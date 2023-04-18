///////////////////////////////////////////////////////////////////////////////
// Course:      ECE8893 - Parallel Programming for FPGAs
// Filename:    conv.h
// Description: Header file for C model simulation of the first 7x7 conv layer 
//              of ResNet-50 DNN
//
// Note:        DO NOT MODIFY THIS CODE!
///////////////////////////////////////////////////////////////////////////////

#ifndef CONV_H_
#define CONV_H_

#include <iostream>

typedef float fm_t;
typedef float wt_t;

// #define IN_FM_DEPTH        3
// #define IN_FM_HEIGHT     736
// #define IN_FM_WIDTH     1280

// #define OUT_FM_DEPTH      64
// #define OUT_FM_HEIGHT    368
// #define OUT_FM_WIDTH     640

#define STRIDE             2
#define PADDING            0 
#define KERNEL_SIZE        5

//--------------------------------------------------------------------------
// Function Declaration
//--------------------------------------------------------------------------
void model_conv (
    fm_t input_feature_map[3][736][1280],
    wt_t layer_weights[64][3][7][7],
    wt_t layer_bias[64],
    fm_t output_feature_map[64][368][640]
);

//KENTA'S CODE
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

void max_pooling1(
    fm_t Y_buf[1][32][187],
    fm_t Y_maxpool_buf[1][32][92]
);

#endif
