import cv2
import numpy as np
import argparse
import os
from PIL import Image

class QM_CODER:
    class TABLE_ITEM:
        def __init__(self, state: int, qc_hex: str, qc_dec: float, inc_state: str, dec_state: str):
            self.state = state
            self.qc_hex = int(qc_hex, 16)
            self.qc_dec = qc_dec
            self.inc_state = inc_state
            self.dec_state = dec_state       

    def __init__(self):
        self.qmtable = []
        self.LPS = '1'
        self.MPS = '0'
        self.state = 0
        self.Qc = 0x59EB
        self.A = 0x10000
        self.C = 0x0000
        self.CT = 11
        self.code_buffer = 0
        self.SC = 0
        self.output = ''
        self.totalbit = 0 # for counting 8 bit-plane
        

    def Load_qm_table(self):
        path = "./qm_table"
        with open(path, 'r') as file:
            for line in file.readlines():
                data = line.split()
                item = QM_CODER.TABLE_ITEM(*data)
                self.qmtable.append(item)

    def SaveEncodedFile(self, target_path, bit_plane_index):
        if bit_plane_index == -1:
            # print("=> len of output: ", len(self.output))
            filename = (target_path.split('/')[-1]).split('.')[0]
        else: # original data
            self.totalbit = self.totalbit + len(self.output)
            filename = (target_path.split('/')[-1]).split('.')[0] + "_bitplane_" + str(bit_plane_index)

        # padding to less than 8 bits
        padbitnum = 8 - len(self.output) % 8
        padbit = ''
        for i in range(padbitnum):
            padbit+=''
        self.output = self.output + padbit

        # use bytearray() to write file
        bytecode = bytearray()
        for i in range(0, len(self.output), 8):
            byte = self.output[i:i+8]
            bytecode.append(int(byte,2))
        
        path = "./Output/QM/" + filename
        with open(path, 'wb') as file:
            file.write(bytecode)

        print(f"file : {filename} Bit-stream len: {len(self.output)}")
        
        if bit_plane_index == 7:
            print(f"Total Bit-stream len: {self.totalbit}")

        self.output = ''
        
    def Encoder(self, pixel_values, target_path, bit_plane_index):
        # np.savetxt('./Debug/QM/pixel_values.txt', pixel_values, fmt='%d')
        for pixel in pixel_values:
            self.A = self.A - self.Qc
            if pixel == int(self.MPS):
                if self.A < 0x8000:
                    if self.A < self.Qc:
                        self.C = self.C + self.A
                        self.A = self.Qc
                    self.Estimate("MPS")
                    self.Renorm_e()
            else: # pixel is LPS
                if self.A >= self.Qc:
                    self.C = self.C + self.A
                    self.A = self.Qc
                self.Estimate("LPS")
                self.Renorm_e()

        self.SaveEncodedFile(target_path, bit_plane_index)
                  
    def Estimate(self, arg):
        # find current qc_hex in table
        for item in self.qmtable:
            if item.qc_hex == self.Qc:
                if arg == "MPS":
                    step = item.inc_state
                if arg == "LPS":
                    step = item.dec_state
            
            if step == 'S':
                self.LPS, self.MPS = self.MPS, self.LPS
            else:
                self.state = item.state + step
                self.Qc = item.qc_hex
            return

    def Renorm_e(self):
        while self.A < 0x8000:
            self.A <<= 1
            self.C <<= 1
            self.CT = self.CT - 1

            if self.CT == 0:
                self.Byte_out()
                self.CT = 8

    def Byte_out(self):
        t = self.C >> 19
        if t > 0xFF:
            self.code_buffer = self.code_buffer + 1 
            if self.code_buffer == 0xFF: 
                self.output = self.output + bin(self.code_buffer)[2:]
                self.code_buffer = 0  
            while self.SC > 0:
                self.output = self.output + bin(self.code_buffer)[2:]
                self.code_buffer = 0 
                self.SC = self.SC - 1
            self.output = self.output + bin(self.code_buffer)[2:]
            self.code_buffer = t
        else:
            if t == 0xFF:
                self.SC = self.SC + 1
            else: # case: t < 0xFF
                while self.SC > 0:
                    self.output = self.output + bin(self.code_buffer)[2:]
                    self.code_buffer = 0xFF 
                    self.output = self.output + bin(self.code_buffer)[2:]
                    self.code_buffer = 0 
                    self.SC = self.SC - 1
                self.output = self.output + bin(self.code_buffer)[2:]
                self.code_buffer = t

        self.C = self.C & 0x7FFFF           

def Gray_code(image_array):
    '''input: integer pixel value / output: 'str'
    first transform to binary format
    and apply 'xor' transform, except the first bit'''

    binary_rep = format(image_array, '08b')  # Convert value to 8-bit binary representation
    binary_list = list(binary_rep)
    transform_list = list()
    transform_list = [0] * 8
    transform_list[0] = binary_list[0]
    for i in range(7):
        transform_list[i+1] = str(int(binary_list[i]) ^ int(binary_list[i+1]))
    
    return ''.join(transform_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("-graycode", action="store_const", const=True, default=False)
    args = parser.parse_args()
    path = args.path
    isGraycode = args.graycode

    qm_code = QM_CODER()
    qm_code.Load_qm_table()

    assert os.path.exists(path) == True

    if isGraycode:
        # gray code
        if 'lena.png' in path or 'baboon.png' in path:
            image = Image.open(path)
            image_array = np.array(image)

            transformed_image_array = np.vectorize(Gray_code)(image_array) # transformed_image_array type: str array
            # print(f"transformed_image_array shape: {transformed_image_array.shape}") # 256*256

            for i in range(8):
                # Extract the ith bit from all string arrays and convert it into a 2D array
                bit_plane = np.array([[int(s[i]) for s in row] for row in transformed_image_array])
                qm_code.Encoder(bit_plane.flatten(), path, i)
    else:
        if '_b' in path or 'halftone' in path:
            image = Image.open(path)
            image_array = np.array(image).flatten()
            qm_code.Encoder(image_array, path, -1)
        
        if 'lena.png' in path or 'baboon.png' in path:
            image = Image.open(path)
            image_array = np.array(image)
            binary_image_array = np.unpackbits(image_array)
            bit_planes = np.split(binary_image_array, 8, axis=0)
            for i, bit_plane in enumerate(bit_planes):
                # print(f"Index {i} array shape: {bit_plane.shape}")
                qm_code.Encoder(bit_plane, path, i)







