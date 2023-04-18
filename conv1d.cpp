#include "conv.h"

void conv_1d_start ( //conv1 code
    fm_t Y_buf[1][32][187], //ignore first dimension, 32 output channels, 187 signal length
    fm_t X_buf[1][1][187], //ignore first dimension, 1 input channel, signal length
    wt_t W_buf[32][1][5], //32 out channels, kernel size:5
    wt_t B_buf[32] //32 outchannels
)
{
    // expect input of size 1, 1, 187
    // padding 2 for kernel size 5
    for (int c = 0; c < 32; c++)
    { // output channels
        for (int l = 0; l < 187; l++)
        { // signal length
            for (int k = 0; k < 5; k++)
            {
                if (l + k < 2 || l + k > 188)
                {
                    Y_buf[0][c][l] += 0;
                }
                else
                {
                    Y_buf[0][c][l] += X_buf[0][0][l + k - 2] * W_buf[c][0][k];
                }
            }
            Y_buf[0][c][l] += B_buf[c];

            // Add ReLU
            if (Y_buf[0][c][l] < 0)
            {
                Y_buf[0][c][l] = 0; // Set to zero negative values
            }
            else
            {
                Y_buf[0][c][l] = Y_buf[0][c][l]; // Sum up all channels in the same output
            }
        }
    }
}

void conv_1d ( //conv2, conv3 code
    fm_t Y_buf[1][32][187], //32 output channels, 187 signal length
    fm_t X_buf[1][32][187], //input channels, signal length
    wt_t W_buf[32][32][5], //32 out channels, kernel size:5
    wt_t B_buf[32] //32 outchannels
)
{
//expects input of size 1, 32, 187
//padding 2 for kernel size 5   
for(int c = 0; c < 32; c++) { //output channels
    for(int l = 0; l < 187; l++) { //signal length
        for(int k = 0; k < 5; k++) {
            for(int x = 0; x < 32; x++) { //input channels
                if(l+k < 2 || l+k > 188) {
                    Y_buf[0][c][l] += 0;
                } else {
                    Y_buf[0][c][l] += X_buf[0][x][l+k-2] * W_buf[c][x][k];
                }
            }
        }
        Y_buf[0][c][l] += B_buf[c];
        
        // Add ReLU
            if (Y_buf[0][c][l] < 0)
            {
                Y_buf[0][c][l] = 0; // Set to zero negative values
            }
            else
            {
                Y_buf[0][c][l] = Y_buf[0][c][l]; // Sum up all channels in the same output
            }
    }
}

}

void max_pooling1(
    fm_t Y_buf[1][32][187],
    fm_t Y_maxpool_buf[1][32][92])
{
// Perform maxpooling
int output_size = (187 - KERNEL_SIZE) / STRIDE + 1;
float max_val = -10000000;
//std::cout << "Max value:  " << max_val << std::endl;
for (int j = 0; j < 32; j++) //32
{ // loop over channels
    for (int k = 0; k < output_size; k++)
    { // loop over output width
            //std::cout << "Window:  " << k << std::endl;
            max_val = -10000000;
                      // Compute max value in kernel window
                      for (int l = 0; l < KERNEL_SIZE; l++)
            { // loop over kernel size
                int idx = k * STRIDE + l;
                float val = Y_buf[0][j][idx];
                //std::cout << "Value:  " << val << std::endl;
                if (val > max_val)
                {
                max_val = val;
                //std::cout << "Max value:  " << max_val << std::endl;
                }
            }
            // Store max value in output tensor
            //std::cout << "Stored value in array:  " << max_val << std::endl;
            Y_maxpool_buf[0][j][k] = max_val;
    }
}
}

// #include "conv.h"

// void model_conv(
//     fm_t input_feature_map[3][736][1280],
//     wt_t layer_weights[64][3][7][7],
//     wt_t layer_bias[64],
//     fm_t output_feature_map[64][368][640])
// {
//     // Input: 736x1280
//     // Output: 368x640 (floor((IH - KH + PaddingLeft + PaddingRight) / 2) + 1 x floor((IW + PaddingLeft + PaddingRight - KW) / 2) + 1)

//     for (int f = 0; f < 64; f++)
//     { // Output depth: 64
//         for (int i = 0; i < 368; i++)
//         { // Output height: 368
//             for (int j = 0; j < 640; j++)
//             { // Output width: 640
//                 fm_t sum = layer_bias[f]; //Add bias
//                 for (int c = 0; c < 3; c++)
//                 { // Channels: 3 (RGB)
//                     for (int kh = 0; kh < 7; kh++)
//                     { // Kernel height: 7
//                         for (int kw = 0; kw < 7; kw++)
//                         {                            // Kernel width: 7
//                             int ii = i * 2 + kh - 3; // Position of the input image with stride 2 and kernel: floor(KH/2)
//                             int jj = j * 2 + kw - 3; // Position of the input image with stride 2 and kernel: floor(KW/2)

//                             // Check if the input position is not padded
//                             if (ii >= 0 && jj >= 0 && ii < 736 && jj < 1280)
//                             {
//                                 sum += input_feature_map[c][ii][jj] * layer_weights[f][c][kh][kw];
//                                 //std::cout << sum << std::endl;
//                             }
                            
//                         }
//                     }
//                     //ReLU
//                     if (sum < 0)
//                     {
//                         output_feature_map[f][i][j] = 0; //Set to zero negative values
//                     }
//                     else
//                     {
//                         output_feature_map[f][i][j] = sum; //Sum up all channels in the same output
//                     }
//                 }
//             }
//         }
//     }
// }
