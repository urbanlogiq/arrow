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

//! Defines boolean kernels on Arrow `BooleanArray`'s, e.g. `AND`, `OR` and `NOT`.
//!
//! These kernels can leverage SIMD if available on your system.  Currently no runtime
//! detection is provided, you should enable the specific SIMD intrinsics using
//! `RUSTFLAGS="-C target-feature=+avx2"` for example.  See the documentation
//! [here](https://doc.rust-lang.org/stable/core/arch/) for more information.

use std::ops::Not;
use std::sync::Arc;

use crate::array::{Array, ArrayData, BooleanArray, PrimitiveArray};
use crate::bitmap::Bitmap;
use crate::buffer::Buffer;
use crate::compute::kernels::comparison::new_all_set_buffer;
use crate::compute::util::apply_bin_op_to_option_bitmap;
use crate::datatypes::{ArrowNumericType, DataType};
use crate::error::{ArrowError, Result};

/// Helper function to implement binary kernels
fn binary_boolean_kernel<F>(
    left: &BooleanArray,
    right: &BooleanArray,
    op: F,
) -> Result<BooleanArray>
where
    F: Fn(&Buffer, &Buffer) -> Result<Buffer>,
{
    if left.offset() != right.offset() {
        return Err(ArrowError::ComputeError(
            "Cannot apply Bitwise binary op when arrays have different offsets."
                .to_string(),
        ));
    }

    let left_data = left.data();
    let right_data = right.data();
    let null_bit_buffer = apply_bin_op_to_option_bitmap(
        left_data.null_bitmap(),
        right_data.null_bitmap(),
        |a, b| a & b,
    )?;
    let values = op(&left_data.buffers()[0], &right_data.buffers()[0])?;
    let data = ArrayData::new(
        DataType::Boolean,
        left.len(),
        None,
        null_bit_buffer,
        left.offset(),
        vec![values],
        vec![],
    );
    Ok(BooleanArray::from(Arc::new(data)))
}

/// Performs `AND` operation on two arrays. If either left or right value is null then the
/// result is also null.
pub fn and(left: &BooleanArray, right: &BooleanArray) -> Result<BooleanArray> {
    binary_boolean_kernel(&left, &right, |a, b| a & b)
}

/// Performs `OR` operation on two arrays. If either left or right value is null then the
/// result is also null.
pub fn or(left: &BooleanArray, right: &BooleanArray) -> Result<BooleanArray> {
    binary_boolean_kernel(&left, &right, |a, b| a | b)
}

/// Performs unary `NOT` operation on an arrays. If value is null then the result is also
/// null.
pub fn not(left: &BooleanArray) -> Result<BooleanArray> {
    let data = left.data();
    let null_bit_buffer = data.null_bitmap().as_ref().map(|b| b.bits.clone());

    let values = !&data.buffers()[0];
    let data = ArrayData::new(
        DataType::Boolean,
        left.len(),
        None,
        null_bit_buffer,
        left.offset(),
        vec![values],
        vec![],
    );
    Ok(BooleanArray::from(Arc::new(data)))
}

/// Copies original array, setting null bit to true if a secondary comparison boolean array is set to true.
/// Typically used to implement NULLIF.
pub fn nullif<T>(
    left: &PrimitiveArray<T>,
    right: &BooleanArray,
) -> Result<PrimitiveArray<T>>
where
    T: ArrowNumericType,
{
    if left.len() != right.len() {
        return Err(ArrowError::ComputeError(
            "Cannot perform comparison operation on arrays of different length"
                .to_string(),
        ));
    }
    let left_data = left.data();

    // If left has no bitmap, create a new one with all values set for nullity op later
    // left=0 (null)   right=null       output bitmap=null
    // left=0          right=1          output bitmap=null
    // left=1 (set)    right=null       output bitmap=set   (passthrough)
    // left=1          right=1 & comp=true    output bitmap=null
    // left=1          right=1 & comp=false   output bitmap=set
    //
    // Thus: result = left null bitmap & (!right_values | !right_bitmap)
    //              OR left null bitmap & !(right_values & right_bitmap)
    //
    // Do the right expression !(right_values & right_bitmap) first since there are two steps
    // TRICK: convert BooleanArray buffer as a bitmap for faster operation
    let right_combo_buffer = match right.data().null_bitmap() {
        Some(right_bitmap) => {
            (&right.values() & &right_bitmap.bits).ok().map(|b| b.not())
        }
        None => Some(!&right.values()),
    };

    let modified_null_buffer = apply_bin_op_to_option_bitmap(
        left_data.null_bitmap(),
        &right_combo_buffer.map(|buf| Bitmap::from(buf)),
        |a, b| a & b,
    )?;

    // Construct new array with same values but modified null bitmap
    let data = ArrayData::new(
        T::get_data_type(),
        left.len(),
        Some(left.len()),
        modified_null_buffer,
        left.offset(),
        left_data.buffers().to_vec(),
        left_data.child_data().to_vec(),
    );
    Ok(PrimitiveArray::<T>::from(Arc::new(data)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::Int32Array;

    #[test]
    fn test_bool_array_and() {
        let a = BooleanArray::from(vec![false, false, true, true]);
        let b = BooleanArray::from(vec![false, true, false, true]);
        let c = and(&a, &b).unwrap();
        assert_eq!(false, c.value(0));
        assert_eq!(false, c.value(1));
        assert_eq!(false, c.value(2));
        assert_eq!(true, c.value(3));
    }

    #[test]
    fn test_bool_array_or() {
        let a = BooleanArray::from(vec![false, false, true, true]);
        let b = BooleanArray::from(vec![false, true, false, true]);
        let c = or(&a, &b).unwrap();
        assert_eq!(false, c.value(0));
        assert_eq!(true, c.value(1));
        assert_eq!(true, c.value(2));
        assert_eq!(true, c.value(3));
    }

    #[test]
    fn test_bool_array_or_nulls() {
        let a = BooleanArray::from(vec![None, Some(false), None, Some(false)]);
        let b = BooleanArray::from(vec![None, None, Some(false), Some(false)]);
        let c = or(&a, &b).unwrap();
        assert_eq!(true, c.is_null(0));
        assert_eq!(true, c.is_null(1));
        assert_eq!(true, c.is_null(2));
        assert_eq!(false, c.is_null(3));
    }

    #[test]
    fn test_bool_array_not() {
        let a = BooleanArray::from(vec![false, false, true, true]);
        let c = not(&a).unwrap();
        assert_eq!(true, c.value(0));
        assert_eq!(true, c.value(1));
        assert_eq!(false, c.value(2));
        assert_eq!(false, c.value(3));
    }

    #[test]
    fn test_bool_array_and_nulls() {
        let a = BooleanArray::from(vec![None, Some(false), None, Some(false)]);
        let b = BooleanArray::from(vec![None, None, Some(false), Some(false)]);
        let c = and(&a, &b).unwrap();
        assert_eq!(true, c.is_null(0));
        assert_eq!(true, c.is_null(1));
        assert_eq!(true, c.is_null(2));
        assert_eq!(false, c.is_null(3));
    }

    #[test]
    fn test_nullif_int_array() {
        let a = Int32Array::from(vec![Some(15), None, Some(8), Some(1), Some(9)]);
        let comp =
            BooleanArray::from(vec![Some(false), None, Some(true), Some(false), None]);
        let res = nullif(&a, &comp).unwrap();

        assert_eq!(15, res.value(0));
        assert_eq!(true, res.is_null(1));
        assert_eq!(true, res.is_null(2)); // comp true, slot 2 turned into null
        assert_eq!(1, res.value(3));
        // Even though comp array / right is null, should still pass through original value
        assert_eq!(9, res.value(4));
        assert_eq!(false, res.is_null(4)); // comp true, slot 2 turned into null
    }
}
