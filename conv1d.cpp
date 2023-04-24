#include "conv.h"

void conv1d_1 ( //conv1 code
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
                Y_buf[0][c][l] = Y_buf[0][c][l]; 
            }
        }
    }
}

void conv1d_2 ( //conv2 code
    fm_t Y_buf[1][32][92], //32 output channels, 187 signal length
    fm_t X_buf[1][32][92], //input channels, signal length
    wt_t W_buf[32][32][5], //32 out channels, kernel size:5
    wt_t B_buf[32] //32 outchannels
)
{
//expects input of size 1, 32, 92
//padding 2 for kernel size 5   
for(int c = 0; c < 32; c++) { //output channels
    for(int l = 0; l < 92; l++) { //signal length
        for(int k = 0; k < 5; k++) {
            for(int x = 0; x < 32; x++) { //input channels
                if(l+k < 2 || l+k > 93) {
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
                Y_buf[0][c][l] = Y_buf[0][c][l];
            }
    }
}

}

void conv1d_3_4 ( //conv3+4 code
    fm_t Y_buf[1][32][44], //32 output channels, 187 signal length
    fm_t X_buf[1][32][44], //input channels, signal length
    wt_t W_buf[32][32][5], //32 out channels, kernel size:5
    wt_t B_buf[32] //32 outchannels
)
{
//expects input of size 1, 32, 92
//padding 2 for kernel size 5   
for(int c = 0; c < 32; c++) { //output channels
    for(int l = 0; l < 44; l++) { //signal length
        for(int k = 0; k < 5; k++) {
            for(int x = 0; x < 32; x++) { //input channels
                if(l+k < 2 || l+k > 45) {
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
                Y_buf[0][c][l] = Y_buf[0][c][l];
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
fm_t max_val = -10000000;
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
                fm_t val = Y_buf[0][j][idx];
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

void max_pooling2(
    fm_t Y_buf[1][32][92],
    fm_t Y_maxpool_buf[1][32][44])
{
// Perform maxpooling
int output_size = (92 - KERNEL_SIZE) / STRIDE + 1;
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

void max_pooling3(
    fm_t Y_buf[1][32][44],
    fm_t Y_maxpool_buf[1][32][20])
{
// Perform maxpooling
int output_size = (44 - KERNEL_SIZE) / STRIDE + 1;
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

void max_pooling4(
    fm_t Y_buf[1][32][20],
    fm_t Y_maxpool_buf[1][32][8])
{
// Perform maxpooling
int output_size = (20 - KERNEL_SIZE) / STRIDE + 1;
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

void max_pooling5(
    fm_t Y_buf[1][32][8],
    fm_t Y_maxpool_buf[1][32][2])
{
// Perform maxpooling
int output_size = (8 - KERNEL_SIZE) / STRIDE + 1;
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

void flatten(
    fm_t in[1][32][2],
    fm_t out[1][64])
{
    for(int i = 0; i < 32; i++){
        for(int j = 0; j < 2; j++){
            out[0][i*2+j] = in[0][i][j];
        }
    }
}



