// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use datafusion::common::{Result, exec_datafusion_err};
use parquet_variant::{VariantPath, VariantPathElement};

/// Parse Spark-rooted paths and the existing rootless path syntax.
pub(crate) fn parse_variant_path(path: &str) -> Result<VariantPath<'_>> {
    // TODO(#142): unify path dialects after deciding compatibility policy (#71, #141).
    // Rootless paths retain Arrow's escaping rules for existing callers.
    if path.starts_with('$') {
        parse_rooted_path(path)
    } else {
        Ok(VariantPath::try_from(path)?)
    }
}

// Spark VariantPathParser's default (non-legacy) grammar, adapted from:
// https://github.com/apache/spark/blob/046f574fce3ffd93925f7bcdd459221fa11a730d/sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/expressions/variant/variantExpressions.scala
// Keep Arrow's path representation without delegating Spark escaping to Arrow.
fn parse_rooted_path(path: &str) -> Result<VariantPath<'_>> {
    let invalid = || exec_datafusion_err!("invalid Variant path {path:?}");
    let mut remaining = path.strip_prefix('$').ok_or_else(invalid)?;
    let mut elements = Vec::new();
    while !remaining.is_empty() {
        if let Some(key) = remaining.strip_prefix('.') {
            let end = key.find(['.', '[']).unwrap_or(key.len());
            if end == 0 {
                return Err(invalid());
            }
            elements.push(VariantPathElement::field(&key[..end]));
            remaining = &key[end..];
        } else if let Some(body) = remaining.strip_prefix('[') {
            if body.starts_with(['\'', '"']) {
                let quote = body.as_bytes()[0] as char;
                let mut chars = body[1..].char_indices();
                let end = loop {
                    let (i, c) = chars.next().ok_or_else(invalid)?;
                    if c == '\\' {
                        chars.next().ok_or_else(invalid)?;
                    } else if c == quote {
                        break i + 1;
                    }
                };
                remaining = body[end + 1..].strip_prefix(']').ok_or_else(invalid)?;
                elements.push(VariantPathElement::field(unescape_key(
                    &body[1..end],
                    quote,
                )));
            } else {
                let end = body.find(']').ok_or_else(invalid)?;
                let index = &body[..end];
                if index.is_empty() || !index.bytes().all(|b| b.is_ascii_digit()) {
                    return Err(invalid());
                }
                let index = index.parse::<i32>().map_err(|_| invalid())? as usize;
                elements.push(VariantPathElement::index(index));
                remaining = &body[end + 1..];
            }
        } else {
            return Err(invalid());
        }
    }
    Ok(VariantPath::new(elements))
}

fn unescape_key(raw: &str, quote: char) -> String {
    let mut output = String::new();
    let mut remaining = raw;
    while let Some((before, after)) = remaining.split_once('\\') {
        output.push_str(before);
        let c = after
            .chars()
            .next()
            .expect("parser checked escape terminator");
        let mut consumed = c.len_utf8();
        match c {
            'b' => output.push('\x08'),
            'f' => output.push('\x0c'),
            'n' => output.push('\n'),
            'r' => output.push('\r'),
            't' => output.push('\t'),
            '/' | '\\' => output.push(c),
            c if c == quote => output.push(c),
            'u' => {
                // Let serde decode valid Unicode escapes (including surrogate
                // pairs). Spark preserves malformed escapes and lone surrogates.
                let decoded = [11, 5].into_iter().find_map(|n| {
                    let escape = after.get(..n)?;
                    if !escape.as_bytes()[1..5].iter().all(u8::is_ascii_hexdigit)
                        || (n == 11
                            && (!escape.as_bytes()[5..].starts_with(b"\\u")
                                || !escape.as_bytes()[7..].iter().all(u8::is_ascii_hexdigit)))
                    {
                        return None;
                    }
                    let value = serde_json::from_str::<String>(&format!("\"\\{escape}\"")).ok()?;
                    Some((n, value))
                });
                if let Some((n, value)) = decoded {
                    consumed = n;
                    output.push_str(&value);
                } else {
                    output.push_str("\\u");
                }
            }
            c => {
                output.push('\\');
                output.push(c);
            }
        }
        remaining = &after[consumed..];
    }
    output.push_str(remaining);
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rooted_and_legacy_paths() {
        for (text, expected) in [
            ("$", vec![]),
            ("", vec![]),
            (
                "$.a[0]",
                vec![VariantPathElement::field("a"), VariantPathElement::index(0)],
            ),
            (
                "a[0]",
                vec![VariantPathElement::field("a"), VariantPathElement::index(0)],
            ),
            ("$['$']", vec![VariantPathElement::field("$")]),
            ("$['a]b']", vec![VariantPathElement::field("a]b")]),
            ("$['']", vec![VariantPathElement::field("")]),
            (r"$['\u0061']", vec![VariantPathElement::field("a")]),
            (r"$['a\'b']", vec![VariantPathElement::field("a'b")]),
        ] {
            assert_eq!(
                parse_variant_path(text).unwrap().path(),
                &expected,
                "{text}"
            );
        }
    }

    #[test]
    fn malformed_rooted_paths_do_not_fall_back() {
        for text in [
            "$bad",
            "$.",
            "$[",
            "$['a'",
            "$[-1]",
            "$[2147483648]",
            "$[*]",
            "$..a",
        ] {
            assert!(parse_variant_path(text).is_err(), "{text}");
        }
    }
}
