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

void conv1d_1 ( //conv1 code
    fm_t Y_buf[1][32][187], //ignore first dimension, 32 output channels, 187 signal length
    fm_t X_buf[1][1][187], //ignore first dimension, 1 input channel, signal length
    wt_t W_buf[32][1][5], //32 out channels, kernel size:5
    wt_t B_buf[32] //32 outchannels
);

void conv1d_2 ( //conv2 code
    fm_t Y_buf[1][32][92], //32 output channels, 187 signal length
    fm_t X_buf[1][32][92], //input channels, signal length
    wt_t W_buf[32][32][5], //32 out channels, kernel size:5
    wt_t B_buf[32] //32 outchannels
);

void conv1d_3_4 ( //conv3+4 code
    fm_t Y_buf[1][32][44], //32 output channels, 187 signal length
    fm_t X_buf[1][32][44], //input channels, signal length
    wt_t W_buf[32][32][5], //32 out channels, kernel size:5
    wt_t B_buf[32] //32 outchannels
);



void max_pooling1(
    fm_t Y_buf[1][32][187],
    fm_t Y_maxpool_buf[1][32][92]
);

void max_pooling2(
    fm_t Y_buf[1][32][92],
    fm_t Y_maxpool_buf[1][32][44]);

void max_pooling3(
    fm_t Y_buf[1][32][44],
    fm_t Y_maxpool_buf[1][32][20]);

void max_pooling4(
    fm_t Y_buf[1][32][20],
    fm_t Y_maxpool_buf[1][32][8]);

void max_pooling5(
    fm_t Y_buf[1][32][8],
    fm_t Y_maxpool_buf[1][32][2]);

void flatten(
    fm_t in[1][32][2],
    fm_t out[1][64]);

void dense1(
    fm_t fixp_conv_layer_output_feature_flat[1][64],
    fm_t fixp_dense1_bias[32],
    fm_t fixp_dense1_weights[32][64],
    fm_t fixp_dense1_output[1][32]);

void dense2(
    fm_t fixp_dense1_output[1][32],
    fm_t fixp_dense2_bias[5],
    fm_t fixp_dense2_weights[5][32],
    fm_t fixp_dense2_output[1][5]);

void tiled_conv (
    fm_t input_feature_map[1][1][187],
    wt_t fixp_conv_layer_weights1[32][1][5],
    wt_t fixp_conv_layer_bias1[32],
    wt_t fixp_conv_layer_weights2[32][32][5],
    wt_t fixp_conv_layer_bias2[32],
    wt_t fixp_conv_layer_weights3[32][32][5],
    wt_t fixp_conv_layer_bias3[32],
    wt_t fixp_conv_layer_weights4[32][32][5],
    wt_t fixp_conv_layer_bias4[32],

    // declare weights and biases for dense1 and dense2
    wt_t fixp_dense1_weights[32][64], 
    wt_t fixp_dense1_bias[32],
    wt_t fixp_dense2_weights[5][32],
    wt_t fixp_dense2_bias[5],
    fm_t conv_layer_output_feature[1][5]
);

#endif
