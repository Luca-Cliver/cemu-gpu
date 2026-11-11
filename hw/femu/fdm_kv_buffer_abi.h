#ifndef CEMU_FDM_KV_BUFFER_ABI_H
#define CEMU_FDM_KV_BUFFER_ABI_H

#include <stdint.h>
#include <stddef.h>

#define CEMU_FDM_KV_BUFFER_VERSION 4U
#define CEMU_FDM_KV_MAX_INPUT_SLOTS 2U
#define CEMU_FDM_KV_MAX_DATA_MRS 126U
#define CEMU_FDM_KV_KIND_COUNT 3U
#define CEMU_FDM_KV_INVALID_INDEX 0U

#define CEMU_FDM_KV_META_MR 0U
#define CEMU_FDM_KV_DATA_MR 1U

#define CEMU_FDM_KV_BIND_MAGIC 0x31424B46U
#define CEMU_FDM_KV_BIND_VERSION 1U
#define CEMU_FDM_KV_BIND_SLOTS_VERSION 2U
#define CEMU_FDM_KV_TRANSFER_PROTOCOL 1U
#define CEMU_FDM_KV_BIND_SELECTOR 2U

#define CEMU_FDM_KV_PACKET_MAGIC 0x31564B46U
#define CEMU_FDM_KV_PACKET_VERSION 1U
#define CEMU_FDM_KV_LAYER_PACKET_VERSION 2U
#define CEMU_FDM_KV_PACKET_HEADER_BYTES 512U
#define CEMU_FDM_KV_RECEIVE_UNIT_BYTES 4U

enum FDMKVOperation {
    FDM_KV_OP_APPEND = 1,
    FDM_KV_OP_DRAIN = 2
};

enum FDMKVReceiveStatus {
    FDM_KV_RX_EMPTY = 0,
    FDM_KV_RX_RECEIVING = 1,
    FDM_KV_RX_AGGREGATING = 2,
    FDM_KV_RX_FAILED = 3,
    FDM_KV_RX_READY = 4,
    FDM_KV_RX_AGGREGATED = 5
};

enum FDMKVKind {
    FDM_KV_TOKEN_K = 0,
    FDM_KV_CHANNEL_K = 1,
    FDM_KV_TOKEN_V = 2
};

typedef struct FDMKV_Identity
{
    uint32_t controller;
    uint32_t namespace_id;
    uint64_t session_id;
    uint64_t generation;
}FDMKV_Identity_t;

typedef struct FDMKV_Geometry
{
    uint32_t batches;
    uint32_t layers;
    uint32_t heads;
    uint32_t head_dim;

    uint64_t max_tokens;

    uint32_t channels_per_page; //channel-k布局下一页有多少个embedding channel
    uint32_t element_bytes;
    uint32_t page_bytes;
    uint32_t reserved;
}FDMKV_Geometry_t;

typedef struct FDMKV_BindParams {
    uint32_t magic;
    uint32_t version;
    uint32_t nvm_nsid;
    uint32_t batch_pages;
    uint64_t session_id;
    uint64_t generation;
    uint64_t input_bytes;
    FDMKV_Geometry_t geometry;
} FDMKV_BindParams_t;

typedef struct FDMKV_BindSlotsParams {
    FDMKV_BindParams_t base;
    uint32_t input_slots;
    uint32_t transfer_protocol;
} FDMKV_BindSlotsParams_t;

typedef struct FDMKV_Layout
{
    FDMKV_Geometry_t geometry;
    uint32_t token_group_tokens;      //token index下一页可以放多少token
    uint32_t channel_group_tokens;    //channel index下一页可以覆盖多少token
    uint32_t channel_group_pages;     //覆盖一个head全部维度要多少channel-k页面
    uint32_t reserved;

    uint64_t head_count;              //所有(batch, layer, KV head) 实例的总数
    uint64_t pages_per_head;          // 一个 head 在最大序列长度下，三种布局合计需要的逻辑页数
    uint64_t total_pages;
}FDMKV_Layout_t;

typedef struct FDMKV_Coordinate
{/*具体数据逻辑坐标*/
    uint32_t batch;
    uint32_t layer;
    uint32_t head;
    uint32_t kind;
    uint64_t token;
    uint32_t channel;
    uint32_t reserved;
} FDMKV_Coordinate_t;

typedef struct FDMKV_HeadState
{/*每个head的进度*/
    uint64_t accepted;
    uint64_t committed[CEMU_FDM_KV_KIND_COUNT];
    uint32_t readers;
    uint32_t reserved;
} FDMKV_HeadState_t;

typedef struct FDMKV_StagedPage
{//一个暂存页
    uint64_t key;                //根据KV页坐标生成的唯一逻辑页键。用来查询暂存页
    FDMKV_Coordinate_t coordinate;  // 这一页对应的KV起始坐标
    uint32_t valid_tokens;         //当前页已经放多少个有效token
    uint32_t next;                 //下一页索引
    uint32_t hash_next;            //哈希链下一页
    uint32_t ready;
}FDMKV_StagedPage_t;

typedef struct FDMKV_BlockQueue
{//一个domain的代写页队列
    uint32_t first;
    uint32_t last;
    uint32_t count;
    uint32_t reserved;
}FDMKV_BlockQueue_t;

typedef struct FDMKV_Writeback
{
    uint64_t deadline_ns;
    uint32_t first;
    uint32_t count;
    uint32_t next;
    uint32_t active;
}FDMKV_Writeback_t;

typedef struct FDMKV_BufferStats
{
    uint64_t append_commands;
    uint64_t appended_head_tokens;
    uint64_t staged_pages;
    uint64_t write_bytes;
    uint64_t read_bytes;
    uint64_t write_batches;
    uint64_t write_blocks;
    uint64_t partial_blocks;
    uint64_t inflight_blocks;
    uint64_t peak_inflight_blocks;
    uint64_t pending_ready_ns;
}FDMKV_BufferStats_t;


/*包格式约定
偏移 0
├── FDMKV_AppendHeader_t：88 字节
├── 保留区域：补零到 512 字节
├── K 数据
├── V 数据
└── 尾部填充：使整包长度满足写入对齐*/

typedef struct FDMKV_AppendHeader
{
    uint32_t magic;
    uint32_t version;
    uint32_t header_bytes;
    uint32_t operation;

    uint64_t session_id;
    uint64_t generation;
    uint64_t request_id;          //本次会话内包序号

    uint64_t wire_bytes;          //pwrite完整长度
    uint64_t payload_bytes;       //KV长度

    uint32_t batch;
    uint32_t layer;
    uint32_t head_first;
    uint32_t head_count;

    uint64_t token_first;
    uint32_t token_count;
    union {
        uint32_t reserved;
        uint32_t batch_count;
    };
}FDMKV_AppendHeader_t;

typedef struct FDMKV_ReceiveState
{
    uint32_t state;
    uint32_t header_valid;        //本轮包头收齐

    uint64_t next_request_id;     //下一包序号
    uint64_t max_write_ready_ns;

    uint64_t bitmap_bytes;
    uint8_t *received_bitmap;

    FDMKV_AppendHeader_t header;
    uint64_t received_units;
    uint64_t received_end;
    uint32_t aggregate_batch;
    uint32_t aggregate_tokens;
} FDMKV_ReceiveState_t;

typedef struct FDMKV_DataSegment {
    uint8_t *address;
    uint64_t bytes;
    uint32_t first_page;
    uint32_t page_count;
} FDMKV_DataSegment_t;

enum FDMKVInputSlotStatus {
    FDM_KV_SLOT_EMPTY,
    FDM_KV_SLOT_RECEIVING,
    FDM_KV_SLOT_RECEIVED,
    FDM_KV_SLOT_READY,
    FDM_KV_SLOT_CONSUMING,
    FDM_KV_SLOT_FAILED,
};

typedef struct FDMKV_InputSlot {
    uint32_t state;
    uint32_t dma_active;
    uint64_t transfer_id;
    uint64_t wire_bytes;
    uint8_t *input;
    FDMKV_ReceiveState_t *receive;
    uint64_t dma_offset;
    uint64_t dma_bytes;
} FDMKV_InputSlot_t;

typedef struct FDMKVBuffer
{
    uint32_t version;
    uint32_t header_bytes;

    FDMKV_Identity_t identity;
    FDMKV_Layout_t layout;

    uint64_t meta_bytes;
    uint64_t data_bytes;

    uint32_t append_max_tokens;
    uint32_t row_bytes;
    uint32_t batch_pages;
    uint32_t writeback_capacity;

    uint32_t free_writeback;
    uint32_t pool_pages;
    uint32_t free_page;
    uint32_t free_pages;

    uint32_t hash_size;
    uint32_t domains;
    uint32_t nand_channels;
    uint32_t block_pages;

    uint32_t initialized;
    uint32_t draining;
    uint32_t sealed;
    uint32_t failed;
    uint32_t leases;
    uint32_t reserved;

    FDMKV_HeadState_t *head_states;
    FDMKV_StagedPage_t *staged;
    FDMKV_BlockQueue_t *blocks;
    uint32_t *page_hash;
    FDMKV_Writeback_t *writebacks;

    uint8_t *input;
    uint64_t input_bytes;
    uint8_t *staged_data;

    FDMKV_BufferStats_t stats;

    FDMKV_ReceiveState_t *receive;
    uint32_t segment_count;
    uint32_t segment_reserved;
    FDMKV_DataSegment_t *segments;
    uint32_t input_slots;
    uint32_t active_slot;
    FDMKV_InputSlot_t *slots;
    uint64_t *received_heads;
    uint64_t next_receive_id;
    uint64_t next_aggregate_id;
    uint64_t last_transfer_id;
}FDMKV_Buffer_t;

static inline uint8_t *fdm_kv_page_data(const FDMKV_Buffer_t *buffer, uint32_t index)
{
    if (!index || index > buffer->pool_pages) {
        return NULL;
    }
    for (uint32_t segment = 0; segment < buffer->segment_count; segment++) {
        const FDMKV_DataSegment_t *range = &buffer->segments[segment];
        if (index >= range->first_page && index - range->first_page < range->page_count) {
            uint64_t offset = segment ? 0 :
                (buffer->input_bytes * (buffer->input_slots ? buffer->input_slots : 1) +
                 buffer->layout.geometry.page_bytes - 1) /
                buffer->layout.geometry.page_bytes * buffer->layout.geometry.page_bytes;
            return range->address + offset +
                (uint64_t)(index - range->first_page) * buffer->layout.geometry.page_bytes;
        }
    }
    return NULL;
}



#if defined(__cplusplus)
#define FDM_KV_STATIC_ASSERT static_assert
#else
#define FDM_KV_STATIC_ASSERT _Static_assert
#endif

FDM_KV_STATIC_ASSERT(sizeof(void *) == 8,
                     "FDM KV requires 64-bit pointers");

FDM_KV_STATIC_ASSERT(sizeof(FDMKV_Identity_t) == 24,
                     "FDMKVIdentity ABI");
FDM_KV_STATIC_ASSERT(sizeof(FDMKV_Geometry_t) == 40,
                     "FDMKVGeometry ABI");
FDM_KV_STATIC_ASSERT(sizeof(FDMKV_BindParams_t) == 80,
                     "FDMKVBindParams ABI");
FDM_KV_STATIC_ASSERT(sizeof(FDMKV_BindSlotsParams_t) == 88,
                     "FDMKVBindSlotsParams ABI");
FDM_KV_STATIC_ASSERT(offsetof(FDMKV_BindParams_t, geometry) == 40,
                     "FDMKVBindParams geometry offset");
FDM_KV_STATIC_ASSERT(sizeof(FDMKV_Layout_t) == 80,
                     "FDMKVLayout ABI");
FDM_KV_STATIC_ASSERT(sizeof(FDMKV_Coordinate_t) == 32,
                     "FDMKVCoordinate ABI");
FDM_KV_STATIC_ASSERT(sizeof(FDMKV_HeadState_t) == 40,
                     "FDMKVHeadState ABI");
FDM_KV_STATIC_ASSERT(sizeof(FDMKV_StagedPage_t) == 56,
                     "FDMKVStagedPage ABI");
FDM_KV_STATIC_ASSERT(sizeof(FDMKV_BlockQueue_t) == 16,
                     "FDMKVBlockQueue ABI");
FDM_KV_STATIC_ASSERT(sizeof(FDMKV_Writeback_t) == 24,
                     "FDMKVWriteback ABI");
FDM_KV_STATIC_ASSERT(sizeof(FDMKV_BufferStats_t) == 88,
                     "FDMKVBufferStats ABI");
FDM_KV_STATIC_ASSERT(sizeof(FDMKV_DataSegment_t) == 24,
                     "FDMKVDataSegment ABI");
FDM_KV_STATIC_ASSERT(sizeof(FDMKV_Buffer_t) == 424,
                     "FDMKVBuffer ABI");
FDM_KV_STATIC_ASSERT(sizeof(FDMKV_InputSlot_t) == 56,
                     "FDMKVInputSlot ABI");

FDM_KV_STATIC_ASSERT(sizeof(FDMKV_AppendHeader_t) == 88,
                     "FDMKVAppendHeader ABI");

FDM_KV_STATIC_ASSERT(offsetof(FDMKV_AppendHeader_t, token_first) == 72,
                     "FDMKVAppendHeader token_first offset");

FDM_KV_STATIC_ASSERT(sizeof(FDMKV_ReceiveState_t) == 152,
                     "FDMKVReceiveState ABI");

FDM_KV_STATIC_ASSERT(offsetof(FDMKV_ReceiveState_t, header) == 40,
                     "FDMKVReceiveState header offset");

FDM_KV_STATIC_ASSERT(offsetof(FDMKV_Coordinate_t, token) == 16,
                     "FDMKVCoordinate token offset");
FDM_KV_STATIC_ASSERT(offsetof(FDMKV_Buffer_t, head_states) == 200,
                     "FDMKVBuffer head_states offset");
FDM_KV_STATIC_ASSERT(offsetof(FDMKV_Buffer_t, input) == 240,
                     "FDMKVBuffer input offset");
FDM_KV_STATIC_ASSERT(offsetof(FDMKV_Buffer_t, stats) == 264,
                     "FDMKVBuffer stats offset");

FDM_KV_STATIC_ASSERT(offsetof(FDMKV_Buffer_t, receive) == 352,
                     "FDMKVBuffer receive offset");
FDM_KV_STATIC_ASSERT(offsetof(FDMKV_Buffer_t, input_slots) == 376,
                     "FDMKVBuffer input_slots offset");

#undef FDM_KV_STATIC_ASSERT



#endif
