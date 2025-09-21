use arrow::array::builder::{
    GenericStringBuilder, ListBuilder, StringViewBuilder,
};
use arrow::array::cast::AsArray;
use arrow::array::*;
use arrow::error::ArrowError;
use arrow::datatypes::{DataType, Field};
use regex::Regex;

use std::collections::HashMap;
use std::sync::Arc;

macro_rules! process_regexp_array_match {
    ($array:expr, $regex_array:expr, $flags_array:expr, $list_builder:expr) => {
        let mut patterns: HashMap<String, Regex> = HashMap::new();

        let complete_pattern = match $flags_array {
            Some(flags) => Box::new($regex_array.iter().zip(flags.iter()).map(
                |(pattern, flags)| {
                    pattern.map(|pattern| match flags {
                        Some(value) => format!("(?{value}){pattern}"),
                        None => pattern.to_string(),
                    })
                },
            )) as Box<dyn Iterator<Item = Option<String>>>,
            None => Box::new(
                $regex_array
                    .iter()
                    .map(|pattern| pattern.map(|pattern| pattern.to_string())),
            ),
        };

        $array
            .iter()
            .zip(complete_pattern)
            .map(|(value, pattern)| {
                match (value, pattern) {
                    // Required for Postgres compatibility:
                    // SELECT regexp_match('foobarbequebaz', ''); = {""}
                    (Some(_), Some(pattern)) if pattern == *"" => {
                        $list_builder.values().append_value("");
                        $list_builder.append(true);
                    }
                    (Some(value), Some(pattern)) => {
                        let existing_pattern = patterns.get(&pattern);
                        let re = match existing_pattern {
                            Some(re) => re,
                            None => {
                                let re = Regex::new(pattern.as_str()).map_err(|e| {
                                    ArrowError::ComputeError(format!(
                                        "Regular expression did not compile: {e:?}"
                                    ))
                                })?;
                                patterns.entry(pattern).or_insert(re)
                            }
                        };
                        match re.captures(value) {
                            Some(caps) => {
                                let mut iter = caps.iter();
                                if caps.len() > 1 {
                                    iter.next();
                                }
                                for m in iter.flatten() {
                                    $list_builder.values().append_value(m.as_str());
                                }

                                $list_builder.append(true);
                            }
                            None => $list_builder.append(false),
                        }
                    }
                    _ => $list_builder.append(false),
                }
                Ok(())
            })
            .collect::<Result<Vec<()>, ArrowError>>()?;
    };
}

fn regexp_array_match<OffsetSize: OffsetSizeTrait>(
    array: &GenericStringArray<OffsetSize>,
    regex_array: &GenericStringArray<OffsetSize>,
    flags_array: Option<&GenericStringArray<OffsetSize>>,
) -> Result<ArrayRef, ArrowError> {
    let builder: GenericStringBuilder<OffsetSize> = GenericStringBuilder::with_capacity(0, 0);
    let mut list_builder = ListBuilder::new(builder);

    process_regexp_array_match!(array, regex_array, flags_array, list_builder);

    Ok(Arc::new(list_builder.finish()))
}

fn regexp_array_match_utf8view(
    array: &StringViewArray,
    regex_array: &StringViewArray,
    flags_array: Option<&StringViewArray>,
) -> Result<ArrayRef, ArrowError> {
    let builder = StringViewBuilder::with_capacity(0);
    let mut list_builder = ListBuilder::new(builder);

    process_regexp_array_match!(array, regex_array, flags_array, list_builder);

    Ok(Arc::new(list_builder.finish()))
}

fn get_scalar_pattern_flag<'a, OffsetSize: OffsetSizeTrait>(
    regex_array: &'a dyn Array,
    flag_array: Option<&'a dyn Array>,
) -> (Option<&'a str>, Option<&'a str>) {
    let regex = regex_array.as_string::<OffsetSize>();
    let regex = regex.is_valid(0).then(|| regex.value(0));

    if let Some(flag_array) = flag_array {
        let flag = flag_array.as_string::<OffsetSize>();
        (regex, flag.is_valid(0).then(|| flag.value(0)))
    } else {
        (regex, None)
    }
}

fn get_scalar_pattern_flag_utf8view<'a>(
    regex_array: &'a dyn Array,
    flag_array: Option<&'a dyn Array>,
) -> (Option<&'a str>, Option<&'a str>) {
    let regex = regex_array.as_string_view();
    let regex = regex.is_valid(0).then(|| regex.value(0));

    if let Some(flag_array) = flag_array {
        let flag = flag_array.as_string_view();
        (regex, flag.is_valid(0).then(|| flag.value(0)))
    } else {
        (regex, None)
    }
}

macro_rules! process_regexp_match {
    ($array:expr, $regex:expr, $list_builder:expr) => {
        $array
            .iter()
            .map(|value| {
                match value {
                    // Required for Postgres compatibility:
                    // SELECT regexp_match('foobarbequebaz', ''); = {""}
                    Some(_) if $regex.as_str().is_empty() => {
                        $list_builder.values().append_value("");
                        $list_builder.append(true);
                    }
                    Some(value) => match $regex.captures(value) {
                        Some(caps) => {
                            let mut iter = caps.iter();
                            if caps.len() > 1 {
                                iter.next();
                            }
                            for m in iter.flatten() {
                                $list_builder.values().append_value(m.as_str());
                            }
                            $list_builder.append(true);
                        }
                        None => $list_builder.append(false),
                    },
                    None => $list_builder.append(false),
                }
                Ok(())
            })
            .collect::<Result<Vec<()>, ArrowError>>()?
    };
}

fn regexp_scalar_match<OffsetSize: OffsetSizeTrait>(
    array: &GenericStringArray<OffsetSize>,
    regex: &Regex,
) -> Result<ArrayRef, ArrowError> {
    let builder: GenericStringBuilder<OffsetSize> = GenericStringBuilder::with_capacity(0, 0);
    let mut list_builder = ListBuilder::new(builder);

    process_regexp_match!(array, regex, list_builder);

    Ok(Arc::new(list_builder.finish()))
}

fn regexp_scalar_match_utf8view(
    array: &StringViewArray,
    regex: &Regex,
) -> Result<ArrayRef, ArrowError> {
    let builder = StringViewBuilder::with_capacity(0);
    let mut list_builder = ListBuilder::new(builder);

    process_regexp_match!(array, regex, list_builder);

    Ok(Arc::new(list_builder.finish()))
}

/// Extract all groups matched by a regular expression for a given String array.
///
/// Modelled after the Postgres [regexp_match].
///
/// Returns a ListArray of [`GenericStringArray`] with each element containing the leftmost-first
/// match of the corresponding index in `regex_array` to string in `array`
///
/// If there is no match, the list element is NULL.
///
/// If a match is found, and the pattern contains no capturing parenthesized subexpressions,
/// then the list element is a single-element [`GenericStringArray`] containing the substring
/// matching the whole pattern.
///
/// If a match is found, and the pattern contains capturing parenthesized subexpressions, then the
/// list element is a [`GenericStringArray`] whose n'th element is the substring matching
/// the n'th capturing parenthesized subexpression of the pattern.
///
/// The flags parameter is an optional text string containing zero or more single-letter flags
/// that change the function's behavior.
///
/// # See Also
/// * [`regexp_is_match`] for matching (rather than extracting) a regular expression against an array of strings
///
/// [regexp_match]: https://www.postgresql.org/docs/current/functions-matching.html#FUNCTIONS-POSIX-REGEXP
pub fn regexp_match(
    array: &dyn Array,
    regex_array: &dyn Datum,
    flags_array: Option<&dyn Datum>,
) -> Result<ArrayRef, ArrowError> {
    let (rhs, is_rhs_scalar) = regex_array.get();

    if array.data_type() != rhs.data_type() {
        return Err(ArrowError::ComputeError(
            "regexp_match() requires both array and pattern to be either Utf8, Utf8View or LargeUtf8"
                .to_string(),
        ));
    }

    let (flags, is_flags_scalar) = match flags_array {
        Some(flags) => {
            let (flags, is_flags_scalar) = flags.get();
            (Some(flags), Some(is_flags_scalar))
        }
        None => (None, None),
    };

    if is_flags_scalar.is_some() && is_rhs_scalar != is_flags_scalar.unwrap() {
        return Err(ArrowError::ComputeError(
            "regexp_match() requires both pattern and flags to be either scalar or array"
                .to_string(),
        ));
    }

    if flags_array.is_some() && rhs.data_type() != flags.unwrap().data_type() {
        return Err(ArrowError::ComputeError(
            "regexp_match() requires both pattern and flags to be either Utf8, Utf8View or LargeUtf8"
                .to_string(),
        ));
    }

    if is_rhs_scalar {
        // Regex and flag is scalars
        let (regex, flag) = match rhs.data_type() {
            DataType::Utf8View => get_scalar_pattern_flag_utf8view(rhs, flags),
            DataType::Utf8 => get_scalar_pattern_flag::<i32>(rhs, flags),
            DataType::LargeUtf8 => get_scalar_pattern_flag::<i64>(rhs, flags),
            _ => {
                return Err(ArrowError::ComputeError(
                    "regexp_match() requires pattern to be either Utf8, Utf8View or LargeUtf8"
                        .to_string(),
                ));
            }
        };

        if regex.is_none() {
            return Ok(new_null_array(
                &DataType::List(Arc::new(Field::new_list_field(
                    array.data_type().clone(),
                    true,
                ))),
                array.len(),
            ));
        }

        let regex = regex.unwrap();

        let pattern = if let Some(flag) = flag {
            format!("(?{flag}){regex}")
        } else {
            regex.to_string()
        };

        let re = Regex::new(pattern.as_str()).map_err(|e| {
            ArrowError::ComputeError(format!("Regular expression did not compile: {e:?}"))
        })?;

        match array.data_type() {
            DataType::Utf8View => regexp_scalar_match_utf8view(array.as_string_view(), &re),
            DataType::Utf8 => regexp_scalar_match(array.as_string::<i32>(), &re),
            DataType::LargeUtf8 => regexp_scalar_match(array.as_string::<i64>(), &re),
            _ => Err(ArrowError::ComputeError(
                "regexp_match() requires array to be either Utf8, Utf8View or LargeUtf8"
                    .to_string(),
            )),
        }
    } else {
        match array.data_type() {
            DataType::Utf8View => {
                let regex_array = rhs.as_string_view();
                let flags_array = flags.map(|flags| flags.as_string_view());
                regexp_array_match_utf8view(array.as_string_view(), regex_array, flags_array)
            }
            DataType::Utf8 => {
                let regex_array = rhs.as_string();
                let flags_array = flags.map(|flags| flags.as_string());
                regexp_array_match(array.as_string::<i32>(), regex_array, flags_array)
            }
            DataType::LargeUtf8 => {
                let regex_array = rhs.as_string();
                let flags_array = flags.map(|flags| flags.as_string());
                regexp_array_match(array.as_string::<i64>(), regex_array, flags_array)
            }
            _ => Err(ArrowError::ComputeError(
                "regexp_match() requires array to be either Utf8, Utf8View or LargeUtf8"
                    .to_string(),
            )),
        }
    }
}
