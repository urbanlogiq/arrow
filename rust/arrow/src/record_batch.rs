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

//! According to the [Arrow Metadata Specification](https://arrow.apache.org/docs/metadata.html):
//!
//! > A record batch is a collection of top-level named, equal length Arrow arrays
//! > (or vectors). If one of the arrays contains nested data, its child arrays are not
//! > required to be the same length as the top-level arrays.

use std::sync::Arc;

use crate::array::*;
use crate::datatypes::*;
use crate::error::{ArrowError, Result};

/// A batch of column-oriented data
#[derive(Clone)]
pub struct RecordBatch {
    schema: Arc<Schema>,
    columns: Vec<Arc<Array>>,
}

impl RecordBatch {
    /// Creates a `RecordBatch` from a schema and columns
    ///
    /// Expects the following:
    ///  * the vec of columns to not be empty
    ///  * the schema and column data types to have equal lengths and match
    ///  * each array in columns to have the same length
    pub fn try_new(schema: Arc<Schema>, columns: Vec<ArrayRef>) -> Result<Self> {
        // check that there are some columns
        if columns.is_empty() {
            return Err(ArrowError::InvalidArgumentError(
                "at least one column must be defined to create a record batch"
                    .to_string(),
            ));
        }
        // check that number of fields in schema match column length
        if schema.fields().len() != columns.len() {
            return Err(ArrowError::InvalidArgumentError(
                "number of columns must match number of fields in schema".to_string(),
            ));
        }
        // check that all columns have the same row count, and match the schema
        let len = columns[0].data().len();
        for i in 0..columns.len() {
            if columns[i].len() != len {
                return Err(ArrowError::InvalidArgumentError(
                    "all columns in a record batch must have the same length".to_string(),
                ));
            }
            if columns[i].data_type() != schema.field(i).data_type() {
                return Err(ArrowError::InvalidArgumentError(format!(
                    "column types must match schema types, expected {:?} but found {:?} at column index {}",
                    schema.field(i).data_type(),
                    columns[i].data_type(),
                    i)));
            }
        }
        Ok(RecordBatch { schema, columns })
    }

    /// Returns the schema of the record batch
    pub fn schema(&self) -> &Arc<Schema> {
        &self.schema
    }

    /// Number of columns in the record batch
    pub fn num_columns(&self) -> usize {
        self.columns.len()
    }

    /// Number of rows in each column
    pub fn num_rows(&self) -> usize {
        self.columns[0].data().len()
    }

    /// Get a reference to a column's array by index
    pub fn column(&self, i: usize) -> &ArrayRef {
        &self.columns[i]
    }

    /// Get a reference to all columns
    pub fn columns(&self) -> &[ArrayRef] {
        &self.columns[..]
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

/// Definition of record batch reader.
pub trait RecordBatchReader {
    /// Returns schemas of this record batch reader.
    /// Implementation of this trait should guarantee that all record batches returned
    /// by this reader should have same schema as returned from this method.
    fn schema(&mut self) -> SchemaRef;

    /// Returns next record batch.
    fn next_batch(&mut self) -> Result<Option<RecordBatch>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::buffer::*;

    #[test]
    fn create_record_batch() {
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Utf8, false),
        ]);

        let v = vec![1, 2, 3, 4, 5];
        let array_data = ArrayData::builder(DataType::Int32)
            .len(5)
            .add_buffer(Buffer::from(v.to_byte_slice()))
            .build();
        let a = Int32Array::from(array_data);

        let v = vec![b'a', b'b', b'c', b'd', b'e'];
        let offset_data = vec![0, 1, 2, 3, 4, 5, 6];
        let array_data = ArrayData::builder(DataType::Utf8)
            .len(5)
            .add_buffer(Buffer::from(offset_data.to_byte_slice()))
            .add_buffer(Buffer::from(v.to_byte_slice()))
            .build();
        let b = BinaryArray::from(array_data);

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
        let boolean_data = ArrayData::builder(DataType::Boolean)
            .len(4)
            .add_buffer(Buffer::from([12_u8]))
            .build();
        let int_data = ArrayData::builder(DataType::Int32)
            .len(4)
            .add_buffer(Buffer::from([42, 28, 19, 31].to_byte_slice()))
            .build();

        // Construct a value array
        let value_data = ArrayData::builder(DataType::Int32)
            .len(9)
            .add_buffer(Buffer::from(&[0, 1, 2, 3, 4, 5, 6, 7, 8].to_byte_slice()))
            .build();

        // Construct a buffer for value offsets, for the nested array:
        //  [[0, 1, 2], [3, 4, 5], [6, 7], [8]]
        let value_offsets = Buffer::from(&[0, 3, 6, 8, 9].to_byte_slice());

        // Construct a list array from the above two
        let list_data_type = DataType::List(Box::new(DataType::Int32));
        let list_data = ArrayData::builder(list_data_type.clone())
            .len(4)
            .add_buffer(value_offsets.clone())
            .add_child_data(value_data.clone())
            .build();

        let struct_array = StructArray::from(vec![
            (
                Field::new("b", DataType::Boolean, false),
                Arc::new(BooleanArray::from(vec![false, false, true, true]))
                    as Arc<Array>,
            ),
            (
                Field::new("c", DataType::Int32, false),
                Arc::new(Int32Array::from(vec![42, 28, 19, 31])),
            ),
            (
                Field::new("d", DataType::List(Box::new(DataType::Int32)), false),
                Arc::new(ListArray::from(list_data.clone())),
            ),
        ]);

        let batch = RecordBatch::from(&struct_array);
        assert_eq!(3, batch.num_columns());
        assert_eq!(4, batch.num_rows());
        assert_eq!(
            struct_array.data_type(),
            &DataType::Struct(batch.schema().fields().to_vec())
        );
        assert_eq!(batch.column(0).data(), boolean_data);
        assert_eq!(batch.column(1).data(), int_data);
        assert_eq!(batch.column(2).data(), list_data);
    }

    #[test]
    fn create_record_batch_with_list_column() {
        let schema = Schema::new(vec![Field::new(
            "a",
            DataType::List(Box::new(DataType::Int32)),
            false,
        )]);

        // Construct a value array
        let value_data = ArrayData::builder(DataType::Int32)
            .len(8)
            .add_buffer(Buffer::from(&[0, 1, 2, 3, 4, 5, 6, 7].to_byte_slice()))
            .build();

        // Construct a buffer for value offsets, for the nested array:
        //  [[0, 1, 2], [3, 4, 5], [6, 7]]
        let value_offsets = Buffer::from(&[0, 3, 6, 8].to_byte_slice());

        // Construct a list array from the above two
        let list_data_type = DataType::List(Box::new(DataType::Int32));
        let list_data = ArrayData::builder(list_data_type.clone())
            .len(3)
            .add_buffer(value_offsets.clone())
            .add_child_data(value_data.clone())
            .build();
        let a = ListArray::from(list_data);

        let record_batch =
            RecordBatch::try_new(Arc::new(schema), vec![Arc::new(a)]).unwrap();

        assert_eq!(3, record_batch.num_rows());
        assert_eq!(1, record_batch.num_columns());
        assert_eq!(
            &DataType::List(Box::new(DataType::Int32)),
            record_batch.schema().field(0).data_type()
        );
        assert_eq!(3, record_batch.column(0).data().len());
    }

    #[test]
    fn create_record_batch_with_list_column_nulls() {
        let schema = Schema::new(vec![Field::new(
            "a",
            DataType::List(Box::new(DataType::Int32)),
            false,
        )]);

        let values_builder = PrimitiveBuilder::<Int32Type>::new(10);
        let mut builder = ListBuilder::new(values_builder);

        builder.values().append_null().unwrap();
        builder.values().append_null().unwrap();
        builder.append(true).unwrap();
        builder.append(false).unwrap();
        builder.append(true).unwrap();

        // [[null, null], null, []]
        let list_array = builder.finish();

        let record_batch =
            RecordBatch::try_new(Arc::new(schema), vec![Arc::new(list_array)]).unwrap();

        assert_eq!(3, record_batch.num_rows());
        assert_eq!(1, record_batch.num_columns());
        assert_eq!(
            &DataType::List(Box::new(DataType::Int32)),
            record_batch.schema().field(0).data_type()
        );
        assert_eq!(3, record_batch.column(0).data().len());

        assert_eq!(false, record_batch.column(0).is_null(0));
        assert_eq!(true, record_batch.column(0).is_null(1));
        assert_eq!(false, record_batch.column(0).is_null(2));

        let col_as_list_array = record_batch
            .column(0)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();

        assert_eq!(2, col_as_list_array.value(0).len());
        assert_eq!(0, col_as_list_array.value(2).len());

        let sublist_0_val = col_as_list_array.value(0);
        let sublist_0 = sublist_0_val.as_any().downcast_ref::<Int32Array>().unwrap();

        assert_eq!(true, sublist_0.is_null(0));
        assert_eq!(true, sublist_0.is_null(1));
    }
}
