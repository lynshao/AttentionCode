# AttentionCode
This repository is the official implementation of paper [AttentionCode: Ultra-Reliable Feedback Codes for Short-Packet Communications] (https://arxiv.org/abs/2205.14955).

If you find this repository useful, please kindly cite as

@article{AttentionCode,

title={Attentioncode: Ultra-reliable feedback codes for short-packet communications},

author={Shao, Yulin and Ozfatura, Emre and Perotti, Alberto and Popovic, Branislav and Gunduz, Deniz},

journal={IEEE Transactions on Communications},

year={2023}

}

## Requirements

Experiments were conducted on Python 3.8.5. To install requirements:

```setup
pip install -r requirements.txt
```


# Well-trained models

Some well-trained models given. The achieved results are as follows:

Noiseless feedback:
| Feedforwad SNR | Feedback SNR | BLER |
| ------------- | ------------- |  ------------- |
| -0.5 dB  | 100 dB  | 4.33e-6 |
| 0 dB  | 100 dB  | 1.17e-7 |
| 0.5 dB  | 100 dB  | 4.17e-9 |

Noisy feedback:
| Feedforwad SNR | Feedback SNR | BLER |
| ------------- | ------------- |  ------------- |
| 0 dB  | 20 dB  | 1.16e-4 |
| 1 dB  | 20 dB  | 1.92e-7 |

To reproduce the results, please run

python main.py --snr1 [input] --snr2 [input] --train 0 --batchSize 100000


# Performance
Noiseless feedback:

<img width="400" alt="1" src="https://github.com/lynshao/AttentionCode/assets/16360158/de8bd62b-ac03-4fdf-90a0-b38bca550b1a">

Noisy feedback:

<img width="397" alt="2" src="https://github.com/lynshao/AttentionCode/assets/16360158/d57038ef-1794-4fbf-80b9-6b76d137d3ea">



