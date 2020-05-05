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

//! Defines the projection execution plan. A projection determines which columns or expressions
//! are returned from a query. The SQL statement `SELECT a, b, a+b FROM t1` is an example
//! of a projection on table `t1` where the expressions `a`, `b`, and `a+b` are the
//! projection expressions.

use std::sync::{Arc, Mutex};

use crate::error::Result;
use crate::execution::physical_plan::{
    BatchIterator, ExecutionPlan, Partition, PhysicalExpr,
};
use arrow::array::{ArrayRef, StructArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

/// Execution plan for a projection
pub struct ProjectionExec {
    /// The projection expressions
    expr: Vec<Arc<dyn PhysicalExpr>>,
    /// The schema once the projection has been applied to the input
    schema: Arc<Schema>,
    /// The input plan
    input: Arc<dyn ExecutionPlan>,
}

impl ProjectionExec {
    /// Create a projection on an input
    pub fn try_new(
        expr: Vec<Arc<dyn PhysicalExpr>>,
        input: Arc<dyn ExecutionPlan>,
    ) -> Result<Self> {
        let input_schema = input.schema();

        // The columns contained within struct type fields will each have their own physical expr.
        // However, these columns do not have a 1:1 relationship with top-level fields in the schema.
        // In order to assign the correct column types for the projected columns,
        // we must build and use for the projection a schema which contains primitive fields for each column contained in a struct field.
        // The original input schema must be maintained so that the columns associated with each struct field can be zipped back up together into structarrays after the projection is performed
        let mut column_fields = Vec::new();
        for field in input_schema.fields() {
            if let DataType::Struct(inner_fields) = field.data_type() {
                for inner_field in inner_fields.iter() {
                    column_fields.push(inner_field.clone());
                }
            } else {
                column_fields.push(field.clone()); // non-struct fields have a 1:1 relationship with columns in the input schema
            }
        }

        let projection_schema = Schema::new(column_fields);
        let fields: Result<Vec<_>> = expr
            .iter()
            .map(|e| {
                Ok(Field::new(
                    &e.name(),
                    e.data_type(&projection_schema)?,
                    true,
                ))
            })
            .collect();

        let schema = Arc::new(Schema::new(fields?));

        Ok(Self {
            expr: expr.clone(),
            schema,
            input: input.clone(),
        })
    }
}

impl ExecutionPlan for ProjectionExec {
    /// Get the schema for this execution plan
    fn schema(&self) -> Arc<Schema> {
        self.schema.clone()
    }

    /// Get the partitions for this execution plan
    fn partitions(&self) -> Result<Vec<Arc<dyn Partition>>> {
        let partitions: Vec<Arc<dyn Partition>> = self
            .input
            .partitions()?
            .iter()
            .map(|p| {
                let expr = self.expr.clone();
                let projection: Arc<dyn Partition> = Arc::new(ProjectionPartition {
                    schema: self.schema.clone(),
                    expr,
                    input: p.clone() as Arc<dyn Partition>,
                });
                projection
            })
            .collect();

        Ok(partitions)
    }
}

/// Represents a single partition of a projection execution plan
struct ProjectionPartition {
    schema: Arc<Schema>,
    expr: Vec<Arc<dyn PhysicalExpr>>,
    input: Arc<dyn Partition>,
}

impl Partition for ProjectionPartition {
    /// Execute the projection
    fn execute(&self) -> Result<Arc<Mutex<dyn BatchIterator>>> {
        Ok(Arc::new(Mutex::new(ProjectionIterator {
            schema: self.schema.clone(),
            expr: self.expr.clone(),
            input: self.input.execute()?,
        })))
    }
}

/// Projection iterator
struct ProjectionIterator {
    schema: Arc<Schema>,
    expr: Vec<Arc<dyn PhysicalExpr>>,
    input: Arc<Mutex<dyn BatchIterator>>,
}

impl BatchIterator for ProjectionIterator {
    /// Get the schema
    fn schema(&self) -> Arc<Schema> {
        self.schema.clone()
    }

    /// Get the next batch
    fn next(&mut self) -> Result<Option<RecordBatch>> {
        let mut input = self.input.lock().unwrap();
        match input.next()? {
            Some(batch) => {
                let arrays = self
                    .expr
                    .iter()
                    .map(|expr| expr.evaluate(&batch))
                    .collect::<Result<Vec<_>>>()?;

                // For struct-type fields, each struct attribute has its own parquet column and gets evaluated individually.
                // Now that we have evaluated our columns (see comments in evaluate in impl PhysicalExpr for Column)
                // we have to put the columns that belong to structs back together into StructArrays so that the correct schema is returned.

                let mut column_arrays = Vec::new(); // Final columns to be returned from the SQL query. One column per top-level field in the parquet schema.
                let mut array_idx = 0; // We need to keep track of the current index within the evaluated arrays as we traverse the evaluated arrays.
                for field in batch.schema().fields().iter() {
                    match field.data_type() {
                        DataType::Struct(inner_fields) => {
                            // For each struct-type top level field in the batch schema, we need to collect all contained arrays
                            let mut field_array_pairs = Vec::new();
                            for inner_field in inner_fields.iter() {
                                field_array_pairs.push((
                                    // every column associated with this struct field will be combined into a StructArray
                                    inner_field.clone(),
                                    arrays[array_idx].clone(),
                                ));
                                array_idx = array_idx + 1;
                            }
                            let struct_array =
                                Arc::new(StructArray::from(field_array_pairs))
                                    as ArrayRef;
                            column_arrays.push(struct_array); // The final returned column is a StructArray
                        }
                        _ => {
                            // Each non-struct top level field is 1:1 associated with an evaluated array
                            column_arrays.push(arrays[array_idx].clone());
                            array_idx = array_idx + 1;
                        }
                    }
                }

                Ok(Some(RecordBatch::try_new(
                    batch.schema().clone(),
                    column_arrays,
                )?))
            }
            None => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::execution::physical_plan::csv::CsvExec;
    use crate::execution::physical_plan::expressions::Column;
    use crate::test;

    #[test]
    fn project_first_column() -> Result<()> {
        let schema = test::aggr_test_schema();

        let partitions = 4;
        let path = test::create_partitioned_csv("aggregate_test_100.csv", partitions)?;

        let csv = CsvExec::try_new(&path, schema.clone(), true, None, 1024)?;

        let projection = ProjectionExec::try_new(
            vec![Arc::new(Column::new(0, &schema.as_ref().field(0).name()))],
            Arc::new(csv),
        )?;

        assert_eq!("c1", projection.schema.field(0).name().as_str());

        let mut partition_count = 0;
        let mut row_count = 0;
        for partition in projection.partitions()? {
            partition_count += 1;
            let iterator = partition.execute()?;
            let mut iterator = iterator.lock().unwrap();
            while let Some(batch) = iterator.next()? {
                assert_eq!(1, batch.num_columns());
                row_count += batch.num_rows();
            }
        }
        assert_eq!(partitions, partition_count);
        assert_eq!(100, row_count);

        Ok(())
    }
}
