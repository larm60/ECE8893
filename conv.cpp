#include "utils.h"

void conv_1d_start ( //conv1 code
    fm_t Y_buf[1][32][187], //ignore first dimension, 32 output channels, 187 signal length
    fm_t X_buf[1][1][187], //ignore first dimension, 1 input channel, signal length
    wt_t W_buf[32][1][5], //32 out channels, kernel size:5
    wt_t B_buf[32] //32 outchannels
)
{
//expect input of size 1, 1, 187
//padding 2 for kernel size 5   
for(int c = 0; c < 32; c++) { //output channels
    for(int l = 0; l < 187; l++) { //signal length
        for(int k = 0; k < 5; k++) {
            if(l+k < 2 || l+k > 188) {
                Y_buf[0][c][l] += 0;
            } else {
                Y_buf[0][c][l] += X_buf[0][0][l+k-2] * W_buf[c][0][k];
            }
        }
        Y_buf[0][c][l] += B_buf[c];
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
    }
}

}
