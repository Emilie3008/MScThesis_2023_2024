import numpy as np

def XS(energy):
    size = len(energy)
    cross_sec = np.zeros((size,))
    cross_sec[size-5:]= 10/np.sqrt(energy[size-5:]) + 500
    
    cross_sec[size//2+15-10]= cross_sec[size//2+15]+110
    cross_sec[size//2+13-10]= cross_sec[size//2+13]-150
    cross_sec[size//2+11-10]= cross_sec[size//2+11]+160

    cross_sec[size//2+30-10]= cross_sec[size//2+15]+110
    cross_sec[size//2+33-10]= cross_sec[size//2+13]-150
    cross_sec[size//2+35-10]= cross_sec[size//2+11]-160
    cross_sec[size//2+29-10]= cross_sec[size//2+15]-110
    
    cross_sec[size//2+35-10]= cross_sec[size//2+11]-160

    cross_sec[:size-5]= cross_sec[size-5] + np.linspace( 20, 1 ,len(cross_sec[:size-5]))
    cross_sec[size//2-10]= cross_sec[size//2]+122
    cross_sec[size//2+1-10]= cross_sec[size//2+1]+127
    cross_sec[size//2+3-10]= cross_sec[size//2+3]-131
    cross_sec[size//2-2-10]= cross_sec[size//2-2]-141
    cross_sec[size//2-3-10]= cross_sec[size//2]+148
    cross_sec[size//2-1-10]= cross_sec[size//2]+116
    cross_sec[size//2-5-10]= cross_sec[size//2]-111
    cross_sec[size//2+6-10]= cross_sec[size//2]+154
    cross_sec[size//2+7-10]= cross_sec[size//2+1]+191
    cross_sec[size//2+9-10]= cross_sec[size//2+3]-143
    cross_sec[size//2+11-10]= cross_sec[size//2+3]-123

    cross_sec[size//2-12-10]= cross_sec[size//2+3]-199
    cross_sec[size//2-20-10]=cross_sec[size//2-40]+110
    cross_sec[size//2-30-10]=cross_sec[size//2-40]+124
    cross_sec[size//2-38-10]=cross_sec[size//2-40]+211
    cross_sec[size//2-40-10]=cross_sec[size//2-40]-184
    cross_sec[size//2-22-10]=cross_sec[size//2-40]-168
    cross_sec[size//2-25-10]=cross_sec[size//2-40]+193
    cross_sec[size//2-29-10]=cross_sec[size//2-40]-179
    cross_sec[size//2-33-10]=cross_sec[size//2-40]+197
    cross_sec[size//2-39-10]=cross_sec[size//2-40]-175
    cross_sec[size//2-35-10]=cross_sec[size//2-40]+144
    cross_sec[size//2-41-10]=cross_sec[size//2-40]+196
    cross_sec[size//2-45-10]=cross_sec[size//2-40]-147
    cross_sec[size//2-23-10]=cross_sec[size//2-40]+137

    cross_sec[size//2-40-10]= cross_sec[size//2+13]-150
    cross_sec[size//2-31-10]= cross_sec[size//2+13]-121
    cross_sec[size//2-33-10]= cross_sec[size//2+13]+152
    cross_sec[size//2-35-10]= cross_sec[size//2+13]-181
    cross_sec[size//2-39-10]= cross_sec[size//2+13]-89
    cross_sec[size//2-34-10]= cross_sec[size//2+13]-99

    cross_sec[size-55]= 10000000
    cross_sec[size-56]= 10000000
    cross_sec[size-57]= 10000000
    cross_sec[size-58]= 10000000
    cross_sec[size-59]= 10000000000
    cross_sec[size-60]= 10000000000
    cross_sec[size-54]= 10000000000
    cross_sec[size-53]= 10000000000
    cross_sec[size-52]= 10000000000
    cross_sec[size-51]= 10000000000
    cross_sec[size-50]= 10000000000
    cross_sec[size-49]= 10000000000
    cross_sec[size-61]= 10000000000
    cross_sec[size-62]= 10000000000
    cross_sec[size-63]= 10000000000
    cross_sec[size-64]= 10000000000
    cross_sec[size-65]= 10000000000
    cross_sec[size-66]= 10000000000
    cross_sec[size-67]= 10000000000
    return cross_sec


def Linear_XS(energy):
    return -0.005e-2*energy+1000