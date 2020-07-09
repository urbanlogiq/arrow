/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.arrow.memory;

import static org.junit.Assert.assertEquals;

import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

import io.netty.buffer.PooledByteBufAllocatorL;

public class TestEmptyArrowBuf {

  private static final int MAX_ALLOCATION = 8 * 1024;
  private static RootAllocator allocator;

  @BeforeClass
  public static void beforeClass() {
    allocator = new RootAllocator(MAX_ALLOCATION);
  }
  
  /** Ensure the allocator is closed. */
  @AfterClass
  public static void afterClass() {
    if (allocator != null) {
      allocator.close();
    }
  }

  @Test
  public void testEmptyArrowBuf() {
    ArrowBuf buf = new ArrowBuf(ReferenceManager.NO_OP, null,
        1024, new PooledByteBufAllocatorL().empty.memoryAddress());

    buf.getReferenceManager().retain();
    buf.getReferenceManager().retain(8);
    assertEquals(1024, buf.capacity());
    assertEquals(1, buf.getReferenceManager().getRefCount());
    assertEquals(0, buf.getActualMemoryConsumed());

    for (int i = 0; i < 10; i++) {
      buf.setByte(i, i);
    }
    assertEquals(0, buf.getActualMemoryConsumed());
    assertEquals(0, buf.getReferenceManager().getSize());
    assertEquals(0, buf.getReferenceManager().getAccountedSize());
    assertEquals(false, buf.getReferenceManager().release());
    assertEquals(false, buf.getReferenceManager().release(2));
    assertEquals(0, buf.getReferenceManager().getAllocator().getLimit());
    assertEquals(buf, buf.getReferenceManager().transferOwnership(buf, allocator).getTransferredBuffer());
    assertEquals(0, buf.readerIndex());
    assertEquals(0, buf.writerIndex());
    assertEquals(1, buf.refCnt());

    ArrowBuf derive = buf.getReferenceManager().deriveBuffer(buf, 0, 100);
    assertEquals(derive, buf);
    assertEquals(1, buf.refCnt());
    assertEquals(1, derive.refCnt());

    buf.close();

  }

}
