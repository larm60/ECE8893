#ifndef CONV_H_
#define CONV_H_

#include <iostream>

typedef float fm_t;
typedef float wt_t;

#define IN_FM_DEPTH        1
#define IN_FM_HEIGHT       1
#define IN_FM_WIDTH      187

#define OUT_FM_DEPTH       1
#define OUT_FM_HEIGHT     32
#define OUT_FM_WIDTH     187

// #define STRIDE             2
// #define PADDING            3 

#define N_TILE_ROWS (int) (OUT_FM_HEIGHT/OUT_BUF_HEIGHT)
#define N_TILE_COLS (int) (OUT_FM_WIDTH/OUT_BUF_WIDTH)

//--------------------------------------------------------------------------
// Function Declaration
//--------------------------------------------------------------------------
void conv_1d_start ( //conv1 code
    fm_t Y_buf[1][32][187], //ignore first dimension, 32 output channels, 187 signal length
    fm_t X_buf[1][1][187], //ignore first dimension, 1 input channel, signal length
    wt_t W_buf[32][1][5], //32 out channels, kernel size:5
    wt_t B_buf[32] //32 outchannels
);

void conv_1d ( //conv2, conv3 code
    fm_t Y_buf[1][32][187], //32 output channels, 187 signal length
    fm_t X_buf[1][32][187], //input channels, signal length
    wt_t W_buf[32][32][5], //32 out channels, kernel size:5
    wt_t B_buf[32] //32 outchannels
);

#endif
