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

//! A two-dimensional batch of column-oriented data with a defined
//! [schema](crate::datatypes::Schema).

use std::sync::Arc;

use crate::array::*;
use crate::datatypes::*;
use crate::error::{ArrowError, Result};

/// A two-dimensional batch of column-oriented data with a defined
/// [schema](crate::datatypes::Schema).
///
/// A `RecordBatch` is a two-dimensional dataset of a number of
/// contiguous arrays, each the same length.
/// A record batch has a schema which must match its arrays’
/// datatypes.
///
/// Record batches are a convenient unit of work for various
/// serialization and computation functions, possibly incremental.
/// See also [CSV reader](crate::csv::Reader) and
/// [JSON reader](crate::json::Reader).
#[derive(Clone, Debug)]
pub struct RecordBatch {
    schema: SchemaRef,
    columns: Vec<Arc<Array>>,
}

impl RecordBatch {
    /// Creates a `RecordBatch` from a schema and columns.
    ///
    /// Expects the following:
    ///  * the vec of columns to not be empty
    ///  * the schema and column data types to have equal lengths
    ///    and match
    ///  * each array in columns to have the same length
    ///
    /// If the conditions are not met, an error is returned.
    ///
    /// # Example
    ///
    /// ```
    /// use std::sync::Arc;
    /// use arrow::array::Int32Array;
    /// use arrow::datatypes::{Schema, Field, DataType};
    /// use arrow::record_batch::RecordBatch;
    ///
    /// # fn main() -> arrow::error::Result<()> {
    /// let id_array = Int32Array::from(vec![1, 2, 3, 4, 5]);
    /// let schema = Schema::new(vec![
    ///     Field::new("id", DataType::Int32, false)
    /// ]);
    ///
    /// let batch = RecordBatch::try_new(
    ///     Arc::new(schema),
    ///     vec![Arc::new(id_array)]
    /// )?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn try_new(schema: SchemaRef, columns: Vec<ArrayRef>) -> Result<Self> {
        let options = RecordBatchOptions::default();
        Self::validate_new_batch(&schema, columns.as_slice(), &options)?;
        Ok(RecordBatch { schema, columns })
    }

    /// Creates a `RecordBatch` from a schema and columns, with additional options,
    /// such as whether to strictly validate field names.
    ///
    /// See [`RecordBatch::try_new`] for the expected conditions.
    pub fn try_new_with_options(
        schema: SchemaRef,
        columns: Vec<ArrayRef>,
        options: &RecordBatchOptions,
    ) -> Result<Self> {
        Self::validate_new_batch(&schema, columns.as_slice(), options)?;
        Ok(RecordBatch { schema, columns })
    }

    /// Validate the schema and columns using [`RecordBatchOptions`]. Returns an error
    /// if any validation check fails.
    fn validate_new_batch(
        schema: &SchemaRef,
        columns: &[ArrayRef],
        options: &RecordBatchOptions,
    ) -> Result<()> {
        // check that there are some columns
        if columns.is_empty() {
            return Err(ArrowError::InvalidArgumentError(
                "at least one column must be defined to create a record batch"
                    .to_string(),
            ));
        }
        // check that number of fields in schema match column length
        if schema.fields().len() != columns.len() {
            return Err(ArrowError::InvalidArgumentError(format!(
                "number of columns({}) must match number of fields({}) in schema",
                columns.len(),
                schema.fields().len(),
            )));
        }
        // check that all columns have the same row count, and match the schema
        let len = columns[0].data().len();

        // This is a bit repetitive, but it is better to check the condition outside the loop
        if options.match_field_names {
            for (i, column) in columns.iter().enumerate() {
                if column.len() != len {
                    return Err(ArrowError::InvalidArgumentError(
                        "all columns in a record batch must have the same length"
                            .to_string(),
                    ));
                }
                if column.data_type() != schema.field(i).data_type() {
                    return Err(ArrowError::InvalidArgumentError(format!(
                        "column types must match schema types, expected {:?} but found {:?} at column index {}",
                        schema.field(i).data_type(),
                        column.data_type(),
                        i)));
                }
            }
        } else {
            for (i, column) in columns.iter().enumerate() {
                if column.len() != len {
                    return Err(ArrowError::InvalidArgumentError(
                        "all columns in a record batch must have the same length"
                            .to_string(),
                    ));
                }
                if !column
                    .data_type()
                    .equals_datatype(schema.field(i).data_type())
                {
                    return Err(ArrowError::InvalidArgumentError(format!(
                        "column types must match schema types, expected {:?} but found {:?} at column index {}",
                        schema.field(i).data_type(),
                        column.data_type(),
                        i)));
                }
            }
        }

        Ok(())
    }

    /// Returns the [`Schema`](crate::datatypes::Schema) of the record batch.
    pub fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    /// Returns the number of columns in the record batch.
    ///
    /// # Example
    ///
    /// ```
    /// use std::sync::Arc;
    /// use arrow::array::Int32Array;
    /// use arrow::datatypes::{Schema, Field, DataType};
    /// use arrow::record_batch::RecordBatch;
    ///
    /// # fn main() -> arrow::error::Result<()> {
    /// let id_array = Int32Array::from(vec![1, 2, 3, 4, 5]);
    /// let schema = Schema::new(vec![
    ///     Field::new("id", DataType::Int32, false)
    /// ]);
    ///
    /// let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(id_array)])?;
    ///
    /// assert_eq!(batch.num_columns(), 1);
    /// # Ok(())
    /// # }
    /// ```
    pub fn num_columns(&self) -> usize {
        self.columns.len()
    }

    /// Returns the number of rows in each column.
    ///
    /// # Panics
    ///
    /// Panics if the `RecordBatch` contains no columns.
    ///
    /// # Example
    ///
    /// ```
    /// use std::sync::Arc;
    /// use arrow::array::Int32Array;
    /// use arrow::datatypes::{Schema, Field, DataType};
    /// use arrow::record_batch::RecordBatch;
    ///
    /// # fn main() -> arrow::error::Result<()> {
    /// let id_array = Int32Array::from(vec![1, 2, 3, 4, 5]);
    /// let schema = Schema::new(vec![
    ///     Field::new("id", DataType::Int32, false)
    /// ]);
    ///
    /// let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(id_array)])?;
    ///
    /// assert_eq!(batch.num_rows(), 5);
    /// # Ok(())
    /// # }
    /// ```
    pub fn num_rows(&self) -> usize {
        self.columns[0].data().len()
    }

    /// Get a reference to a column's array by index.
    ///
    /// # Panics
    ///
    /// Panics if `index` is outside of `0..num_columns`.
    pub fn column(&self, index: usize) -> &ArrayRef {
        &self.columns[index]
    }

    /// Get a reference to all columns in the record batch.
    pub fn columns(&self) -> &[ArrayRef] {
        &self.columns[..]
    }

    pub fn try_empty(schema: SchemaRef) -> Result<Self> {
        let columns = schema
            .fields()
            .iter()
            .map(|f| -> ArrayRef {
                match f.data_type() {
                    DataType::Boolean => Arc::new(BooleanArray::from(Vec::<bool>::new())),
                    DataType::Int8 => Arc::new(Int8Array::from(Vec::<i8>::new())),
                    DataType::Int16 => Arc::new(Int16Array::from(Vec::<i16>::new())),
                    DataType::Int32 => Arc::new(Int32Array::from(Vec::<i32>::new())),
                    DataType::Int64 => Arc::new(Int64Array::from(Vec::<i64>::new())),
                    DataType::UInt8 => Arc::new(UInt8Array::from(Vec::<u8>::new())),
                    DataType::UInt16 => Arc::new(UInt16Array::from(Vec::<u16>::new())),
                    DataType::UInt32 => Arc::new(UInt32Array::from(Vec::<u32>::new())),
                    DataType::UInt64 => Arc::new(UInt64Array::from(Vec::<u64>::new())),
                    DataType::Float16 => todo!(),
                    DataType::Float32 => Arc::new(Float32Array::from(Vec::<f32>::new())),
                    DataType::Float64 => Arc::new(Float64Array::from(Vec::<f64>::new())),
                    DataType::Date32(_) => Arc::new(Date32Array::from(Vec::<i32>::new())),
                    DataType::Date64(_) => Arc::new(Date64Array::from(Vec::<i64>::new())),
                    DataType::Time32(TimeUnit::Second) => {
                        Arc::new(Time32SecondArray::from(Vec::<i32>::new()))
                    }
                    DataType::Time32(TimeUnit::Millisecond) => {
                        Arc::new(Time32MillisecondArray::from(Vec::<i32>::new()))
                    }
                    DataType::Time32(_) => unimplemented!(),
                    DataType::Time64(TimeUnit::Microsecond) => {
                        Arc::new(Time64MicrosecondArray::from(Vec::<i64>::new()))
                    }
                    DataType::Time64(TimeUnit::Nanosecond) => {
                        Arc::new(Time64NanosecondArray::from(Vec::<i64>::new()))
                    }
                    DataType::Time64(_) => unimplemented!(),
                    DataType::Duration(TimeUnit::Second) => {
                        Arc::new(DurationSecondArray::from(Vec::<i64>::new()))
                    }
                    DataType::Duration(TimeUnit::Millisecond) => {
                        Arc::new(DurationMillisecondArray::from(Vec::<i64>::new()))
                    }
                    DataType::Duration(TimeUnit::Microsecond) => {
                        Arc::new(DurationMicrosecondArray::from(Vec::<i64>::new()))
                    }
                    DataType::Duration(TimeUnit::Nanosecond) => {
                        Arc::new(DurationNanosecondArray::from(Vec::<i64>::new()))
                    }
                    DataType::Interval(IntervalUnit::YearMonth) => {
                        Arc::new(IntervalYearMonthArray::from(Vec::<i32>::new()))
                    }
                    DataType::Interval(IntervalUnit::DayTime) => {
                        Arc::new(IntervalDayTimeArray::from(Vec::<i64>::new()))
                    }
                    DataType::LargeBinary => {
                        Arc::new(LargeBinaryArray::from(Vec::<&[u8]>::new()))
                    }
                    DataType::Binary => Arc::new(BinaryArray::from(Vec::<&[u8]>::new())),
                    DataType::FixedSizeBinary(_) => {
                        Arc::new(FixedSizeBinaryArray::from(Vec::<Vec<u8>>::new()))
                    }
                    DataType::Utf8 => Arc::new(StringArray::from(Vec::<&str>::new())),
                    DataType::LargeUtf8 => {
                        Arc::new(LargeStringArray::from(Vec::<&str>::new()))
                    }
                    DataType::Null => Arc::new(NullArray::new(0)),
                    DataType::FixedSizeList(_, _) => {
                        let array_data = ArrayData::builder(f.data_type().clone())
                            .len(0)
                            .add_buffer(Vec::new().into())
                            .build();
                        Arc::new(FixedSizeListArray::from(array_data))
                    }
                    DataType::LargeList(_) => {
                        let array_data = ArrayData::builder(f.data_type().clone())
                            .len(0)
                            .add_buffer(Vec::new().into())
                            .build();
                        Arc::new(LargeListArray::from(array_data))
                    }
                    DataType::List(_) => {
                        let array_data = ArrayData::builder(f.data_type().clone())
                            .len(0)
                            .add_buffer(Vec::new().into())
                            .build();
                        Arc::new(ListArray::from(array_data))
                    }
                    DataType::Struct(_) => {
                        let array_data =
                            ArrayData::builder(f.data_type().clone()).len(0).build();
                        Arc::new(StructArray::from(array_data))
                    }
                    DataType::Union(_) => {
                        let array_data =
                            ArrayData::builder(f.data_type().clone()).len(0).build();
                        Arc::new(UnionArray::from(array_data))
                    }
                    DataType::Decimal(_, _) => {
                        let array_data = ArrayData::builder(f.data_type().clone())
                            .len(0)
                            .add_buffer(Vec::new().into())
                            .build();
                        Arc::new(DecimalArray::from(array_data))
                    }
                    DataType::Dictionary(_, _) => todo!(),
                    DataType::Timestamp(TimeUnit::Second, tz) => {
                        Arc::new(TimestampSecondArray::from_vec(vec![], tz.clone()))
                    }
                    DataType::Timestamp(TimeUnit::Millisecond, tz) => {
                        Arc::new(TimestampMillisecondArray::from_vec(vec![], tz.clone()))
                    }
                    DataType::Timestamp(TimeUnit::Microsecond, tz) => {
                        Arc::new(TimestampMicrosecondArray::from_vec(vec![], tz.clone()))
                    }
                    DataType::Timestamp(TimeUnit::Nanosecond, tz) => {
                        Arc::new(TimestampNanosecondArray::from_vec(vec![], tz.clone()))
                    }
                }
            })
            .collect::<Vec<_>>();

        Self::try_new(schema, columns)
    }
}

/// Options that control the behaviour used when creating a [`RecordBatch`].
#[derive(Debug)]
pub struct RecordBatchOptions {
    /// Match field names of structs and lists. If set to `true`, the names must match.
    pub match_field_names: bool,
}

impl Default for RecordBatchOptions {
    fn default() -> Self {
        Self {
            match_field_names: true,
        }
    }
}

impl From<&StructArray> for RecordBatch {
    /// Create a record batch from struct array.
    ///
    /// This currently does not flatten and nested struct types
    fn from(struct_array: &StructArray) -> Self {
        if let DataType::Struct(fields) = struct_array.data_type() {
            let schema = Schema::new(fields.clone());
            let columns = struct_array.boxed_fields.clone();
            RecordBatch {
                schema: Arc::new(schema),
                columns,
            }
        } else {
            unreachable!("unable to get datatype as struct")
        }
    }
}

impl Into<StructArray> for RecordBatch {
    fn into(self) -> StructArray {
        self.schema
            .fields
            .iter()
            .zip(self.columns.iter())
            .map(|t| (t.0.clone(), t.1.clone()))
            .collect::<Vec<(Field, ArrayRef)>>()
            .into()
    }
}

/// Trait for types that can read `RecordBatch`'s.
pub trait RecordBatchReader: Iterator<Item = Result<RecordBatch>> {
    /// Returns the schema of this `RecordBatchReader`.
    ///
    /// Implementation of this trait should guarantee that all `RecordBatch`'s returned by this
    /// reader should have the same schema as returned from this method.
    fn schema(&self) -> SchemaRef;

    /// Reads the next `RecordBatch`.
    #[deprecated(
        since = "2.0.0",
        note = "This method is deprecated in favour of `next` from the trait Iterator."
    )]
    fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
        self.next().transpose()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::buffer::Buffer;

    #[test]
    fn create_record_batch() {
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Utf8, false),
        ]);

        let a = Int32Array::from(vec![1, 2, 3, 4, 5]);
        let b = StringArray::from(vec!["a", "b", "c", "d", "e"]);

        let record_batch =
            RecordBatch::try_new(Arc::new(schema), vec![Arc::new(a), Arc::new(b)])
                .unwrap();

        assert_eq!(5, record_batch.num_rows());
        assert_eq!(2, record_batch.num_columns());
        assert_eq!(&DataType::Int32, record_batch.schema().field(0).data_type());
        assert_eq!(&DataType::Utf8, record_batch.schema().field(1).data_type());
        assert_eq!(5, record_batch.column(0).data().len());
        assert_eq!(5, record_batch.column(1).data().len());
    }

    #[test]
    fn create_record_batch_schema_mismatch() {
        let schema = Schema::new(vec![Field::new("a", DataType::Int32, false)]);

        let a = Int64Array::from(vec![1, 2, 3, 4, 5]);

        let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(a)]);
        assert!(!batch.is_ok());
    }

    #[test]
    fn create_record_batch_field_name_mismatch() {
        let struct_fields = vec![
            Field::new("a1", DataType::Int32, false),
            Field::new(
                "a2",
                DataType::List(Box::new(Field::new("item", DataType::Int8, false))),
                false,
            ),
        ];
        let struct_type = DataType::Struct(struct_fields);
        let schema = Arc::new(Schema::new(vec![Field::new("a", struct_type, true)]));

        let a1: ArrayRef = Arc::new(Int32Array::from(vec![1, 2]));
        let a2_child = Int8Array::from(vec![1, 2, 3, 4]);
        let a2 = ArrayDataBuilder::new(DataType::List(Box::new(Field::new(
            "array",
            DataType::Int8,
            false,
        ))))
        .add_child_data(a2_child.data())
        .len(2)
        .add_buffer(Buffer::from(vec![0i32, 3, 4].to_byte_slice()))
        .build();
        let a2: ArrayRef = Arc::new(ListArray::from(a2));
        let a = ArrayDataBuilder::new(DataType::Struct(vec![
            Field::new("aa1", DataType::Int32, false),
            Field::new("a2", a2.data_type().clone(), false),
        ]))
        .add_child_data(a1.data())
        .add_child_data(a2.data())
        .len(2)
        .build();
        let a: ArrayRef = Arc::new(StructArray::from(a));

        // creating the batch with field name validation should fail
        let batch = RecordBatch::try_new(schema.clone(), vec![a.clone()]);
        assert!(batch.is_err());

        // creating the batch without field name validation should pass
        let options = RecordBatchOptions {
            match_field_names: false,
        };
        let batch = RecordBatch::try_new_with_options(schema, vec![a], &options);
        assert!(batch.is_ok());
    }

    #[test]
    fn create_record_batch_record_mismatch() {
        let schema = Schema::new(vec![Field::new("a", DataType::Int32, false)]);

        let a = Int32Array::from(vec![1, 2, 3, 4, 5]);
        let b = Int32Array::from(vec![1, 2, 3, 4, 5]);

        let batch =
            RecordBatch::try_new(Arc::new(schema), vec![Arc::new(a), Arc::new(b)]);
        assert!(!batch.is_ok());
    }

    #[test]
    fn create_record_batch_from_struct_array() {
        let boolean = Arc::new(BooleanArray::from(vec![false, false, true, true]));
        let int = Arc::new(Int32Array::from(vec![42, 28, 19, 31]));
        let struct_array = StructArray::from(vec![
            (
                Field::new("b", DataType::Boolean, false),
                boolean.clone() as ArrayRef,
            ),
            (
                Field::new("c", DataType::Int32, false),
                int.clone() as ArrayRef,
            ),
        ]);

        let batch = RecordBatch::from(&struct_array);
        assert_eq!(2, batch.num_columns());
        assert_eq!(4, batch.num_rows());
        assert_eq!(
            struct_array.data_type(),
            &DataType::Struct(batch.schema().fields().to_vec())
        );
        assert_eq!(batch.column(0).as_ref(), boolean.as_ref());
        assert_eq!(batch.column(1).as_ref(), int.as_ref());
    }
}
