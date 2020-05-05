﻿// Licensed to the Apache Software Foundation (ASF) under one or more
// contributor license agreements. See the NOTICE file distributed with
// this work for additional information regarding copyright ownership.
// The ASF licenses this file to You under the Apache License, Version 2.0
// (the "License"); you may not use this file except in compliance with
// the License.  You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

using Apache.Arrow.Memory;
using Apache.Arrow.Types;
using System;
using System.Collections.Generic;

namespace Apache.Arrow
{
    public class BooleanArray: Array
    {
        public class Builder : IArrowArrayBuilder<bool, BooleanArray, Builder>
        {
            private ArrowBuffer.Builder<byte> ValueBuffer { get; }

            private ArrowBuffer.Builder<byte> ValidityBuffer { get; }

            public int Length { get; protected set; }
            public int Capacity => BitUtility.ByteCount(ValueBuffer.Capacity);
            public int NullCount { get; protected set; }

            public Builder()
            {
                ValueBuffer = new ArrowBuffer.Builder<byte>();
                ValidityBuffer = new ArrowBuffer.Builder<byte>();
                Length = 0;
                NullCount = 0;
            }

            public Builder Append(bool value)
            {
                return NullableAppend(value);
            }

            private Builder NullableAppend(bool? value)
            {
                if (Length % 8 == 0)
                {
                    // append a new byte to the buffer when needed
                    ValueBuffer.Append(0);
                    ValidityBuffer.Append(0);
                }
                BitUtility.SetBit(ValueBuffer.Span, Length, value.GetValueOrDefault());
                BitUtility.SetBit(ValidityBuffer.Span, Length, value.HasValue);
                NullCount += value.HasValue ? 0 : 1;
                Length++;
                return this;
            }

            public Builder Append(ReadOnlySpan<bool> span)
            {
                foreach (var value in span)
                {
                    Append(value);
                }
                return this;
            }

            public Builder AppendRange(IEnumerable<bool> values)
            {
                foreach (var value in values)
                {
                    Append(value);
                }
                return this;
            }

            public Builder AppendNull()
            {
                return NullableAppend(null);
            }

            public BooleanArray Build(MemoryAllocator allocator = default)
            {
                var validityBuffer = NullCount > 0
                                        ? ValidityBuffer.Build(allocator)
                                        : ArrowBuffer.Empty;

                return new BooleanArray(
                    ValueBuffer.Build(allocator), validityBuffer,
                    Length, NullCount, 0);
            }

            public Builder Clear()
            {
                ValueBuffer.Clear();
                Length = 0;
                return this;
            }

            public Builder Reserve(int capacity)
            {
                if (capacity < 0)
                {
                    throw new ArgumentOutOfRangeException(nameof(capacity));
                }

                ValueBuffer.Reserve(BitUtility.ByteCount(capacity));
                ValidityBuffer.Reserve(BitUtility.ByteCount(capacity));
                return this;
            }

            public Builder Resize(int length)
            {
                if (length < 0)
                {
                    throw new ArgumentOutOfRangeException(nameof(length));
                }

                ValueBuffer.Resize(BitUtility.ByteCount(length));
                ValidityBuffer.Resize(BitUtility.ByteCount(length));
                Length = length;
                return this;
            }

            public Builder Toggle(int index)
            {
                CheckIndex(index);
                BitUtility.ToggleBit(ValueBuffer.Span, index);
                BitUtility.SetBit(ValidityBuffer.Span, index, true);
                return this;
            }

            public Builder Set(int index)
            {
                CheckIndex(index);
                BitUtility.SetBit(ValueBuffer.Span, index);
                BitUtility.SetBit(ValidityBuffer.Span, index, true);
                return this;
            }

            public Builder Set(int index, bool value)
            {
                CheckIndex(index);
                BitUtility.SetBit(ValueBuffer.Span, index, value);
                BitUtility.SetBit(ValidityBuffer.Span, index, true);
                return this;
            }

            public Builder Swap(int i, int j)
            {
                CheckIndex(i);
                CheckIndex(j);
                var bi = BitUtility.GetBit(ValueBuffer.Span, i);
                var biValid = BitUtility.GetBit(ValidityBuffer.Span, i);
                var bj = BitUtility.GetBit(ValueBuffer.Span, j);
                var bjValid = BitUtility.GetBit(ValidityBuffer.Span, j);
                BitUtility.SetBit(ValueBuffer.Span, i, bj);
                BitUtility.SetBit(ValidityBuffer.Span, i, bjValid);
                BitUtility.SetBit(ValueBuffer.Span, j, bi);
                BitUtility.SetBit(ValidityBuffer.Span, j, biValid);
                return this;
            }

            private void CheckIndex(int index)
            {
                if (index < 0 || index >= Length)
                {
                    throw new ArgumentOutOfRangeException(nameof(index));
                }
            }
        }

        public ArrowBuffer ValueBuffer => Data.Buffers[1];
        public ReadOnlySpan<byte> Values => ValueBuffer.Span.Slice(0, (int) Math.Ceiling(Length / 8.0));

        public BooleanArray(
            ArrowBuffer valueBuffer, ArrowBuffer nullBitmapBuffer,
            int length, int nullCount, int offset)
            : this(new ArrayData(BooleanType.Default, length, nullCount, offset,
                new[] { nullBitmapBuffer, valueBuffer }))
        { }

        public BooleanArray(ArrayData data)
            : base(data)
        {
            data.EnsureDataType(ArrowTypeId.Boolean);
        }

        public override void Accept(IArrowArrayVisitor visitor) => Accept(this, visitor);

        [Obsolete("GetBoolean does not support null values. Use GetValue instead (which this method invokes internally).")]
        public bool GetBoolean(int index)
        {
            return GetValue(index).GetValueOrDefault();
        }

        public bool? GetValue(int index)
        {
            return IsNull(index)
                ? (bool?)null
                : BitUtility.GetBit(ValueBuffer.Span, index + Offset);
        }
    }
}
