// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::any::Any;

use arrow::array::{Array, ArrayRef, AsArray, new_empty_array};
use arrow::compute::kernels::regexp;
use datafusion_common::{arrow_datafusion_err, DataFusionError};
use arrow::datatypes::{
    DataType, Int32Type
};
use datafusion_common::{exec_err, internal_err, Result};
use datafusion_expr::{
    ColumnarValue, ScalarUDFImpl, Signature, TypeSignature, Volatility,
};
use datafusion_functions::utils::make_scalar_function;

/// https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.functions.regexp_extract.html
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkRegexpExtract {
    signature: Signature,
}

impl Default for SparkRegexpExtract {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkRegexpExtract {
    pub fn new() -> Self {
        use DataType::*;
        Self {
            signature: Signature::one_of(
                vec![
                    // Planner attempts coercion to the target type starting with the most preferred candidate.
                    // For example, given input `(Utf8View, Utf8)`, it first tries coercing to `(Utf8View, Utf8View)`.
                    // If that fails, it proceeds to `(Utf8, Utf8)`.
                    TypeSignature::Exact(vec![Utf8View, Utf8View, Int32]),
                    TypeSignature::Exact(vec![Utf8, Utf8, Int32]),
                    TypeSignature::Exact(vec![LargeUtf8, LargeUtf8, Int32]),
                ],
                Volatility::Immutable,
            ),
        }
    }
}

impl ScalarUDFImpl for SparkRegexpExtract {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "regexp_match"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        match &arg_types[0] {
            DataType::Utf8View => Ok(DataType::Utf8View),
            DataType::Utf8 => Ok(DataType::Utf8),
            DataType::LargeUtf8 => Ok(DataType::LargeUtf8),
            _ => {
                exec_err!(
                    "Spark `regexp_extract` function: argument must be Utf8, Utf8View or LargeUtf8, got {:?}",
                    arg_types[0]
                )
            }
        }
    }

    fn invoke_with_args(
        &self,
        args: datafusion_expr::ScalarFunctionArgs,
    ) -> Result<ColumnarValue> {
        make_scalar_function(spark_regexp_extract, vec![])(&args.args)
    }
}

fn spark_regexp_extract(args: &[ArrayRef]) -> Result<ArrayRef> {
    if args.len() != 3 {
        return internal_err!(
            "Spark `regexp_extract` function requires 3 argument, got {}",
            args.len()
        );
    };

    let regex_matches = regexp::regexp_match(&args[0], &args[1], None)
        .map_err(|e| arrow_datafusion_err!(e))?;
    let idx = args[2].as_primitive::<Int32Type>();
    let idx = idx.value(0) as usize;

    if regex_matches.len() < idx {
        // fmt.Println("Warning: idx {} is out of bounds, returning empty array", idx);
        return Ok(new_empty_array(args[0].data_type()));
    }

    Ok(regex_matches.slice(idx - 1, 1))
}

#[cfg(test)]
mod tests {
    use crate::function::regex::regexp_extract::spark_regexp_extract;
    use arrow::array::StringArray;
    use arrow::array::{GenericStringBuilder, ListBuilder, Int32Array};
    use std::sync::Arc;

    #[test]
    fn test_spark_regexp_extract() {
        let values = StringArray::from(vec!["abc"; 5]);
        let patterns =
            StringArray::from(vec!["^(a)", "^(a)", "(b|d)", "(B|D)", "^(b|c)"]);

        let elem_builder: GenericStringBuilder<i32> = GenericStringBuilder::new();
        let mut expected_builder = ListBuilder::new(elem_builder);
        expected_builder.values().append_value("a");
        expected_builder.append(true);
        expected_builder.append(false);
        expected_builder.values().append_value("b");
        expected_builder.append(true);
        expected_builder.append(false);
        expected_builder.append(false);
        let expected = expected_builder.finish();

        let idx = Int32Array::new(vec![1, 2, 3, 4].into(), None);
        let re = spark_regexp_extract(&[Arc::new(values), Arc::new(patterns), Arc::new(idx)]).unwrap();

        assert_eq!(re.as_ref(), &expected);
    }
}
