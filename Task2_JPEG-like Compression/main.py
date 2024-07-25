# run: python main.py '.\Test Images\GrayImages\Baboon.raw' 50
import numpy as np
from io import StringIO
import argparse
import itertools
import cv2
import csv
import math
import matplotlib.pyplot as plt
from skimage.transform import resize

OUTPUT_PATH = "./Output/"

quantization_matrix_l = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61], 
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56], 
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77], 
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101], 
        [72, 92, 95, 98, 112, 100, 103, 99]
    ]
)

quantization_matrix_c = np.array(
    [
        [17, 18, 24, 47, 99, 99, 99, 99], 
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99], 
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99], 
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99], 
        [99, 99, 99, 99, 99, 99, 99, 99]
    ]
)

zigZagMartixOrder = np.array(
    [
        [0,1,5,6,14,15,27,28],
        [2,4,7,13,16,26,29,42],
        [3,8,12,17,25,30,41,43],
        [9,11,18,24,31,40,44,53],
        [10,19,23,32,39,45,52,54],
        [20,22,33,38,46,51,55,60],
        [21,34,37,47,50,56,59,61],
        [35,36,48,49,57,58,62,63]
    ]
)

dc_table = [
    "00","010","011","100","101","110","1110","11110",
    "111110", "1111110","11111110","111111110"
]

# For debugging
def Write_to_file(filename, targetList):
    filename = OUTPUT_PATH + filename
    with open(filename, 'w') as file:
        for item in targetList:
            for value in item:
                file.write(str(value) + ' ')
            file.write('\n')

class ENCODER:
    def __init__(self, qf) -> None:
        self.ac_luminance_table = {}
        self.print_once = 0
        self.qf = qf
        self.block_size = 8
        self.quantizedBlocks = [] # output after quantization
        self.precodes = []
        self.LoadTable()
        self.encode_codeword = "" # output of the encoder

    def Preprocess(self, img):
        row , col = img.shape
        img = img - 128.

        split_blocks = []
        dct_blocks = []

        # split image into 8*8 block
        for j in range(0, row, self.block_size):
            for i in range(0, col, self.block_size):
                block = img[j:j+self.block_size, i:i+self.block_size]
                split_blocks.append(block)
        
        # dct
        for block in split_blocks:
            dct_block = cv2.dct(np.float32(block))
            dct_blocks.append(dct_block)

        if self.qf < 50:
            factor = 5000 / self.qf
        else:
            factor = 200 - (2 * self.qf)

        quantizationMatrix = np.floor(quantization_matrix_l * factor / 100).astype(np.int16)
        quantizationMatrix[quantizationMatrix==0] = 1

        for block in dct_blocks:
            quantized_block = np.round(block / quantizationMatrix).astype(np.int16)
            self.quantizedBlocks.append(quantized_block)


    def RunLengthCoding(self, array):
        '''
        1. add EOB
        2. encode
        '''
        # add EOB
        if array[-1] != 0:
            array.append('EOB')
        elif array == [0]*64:
            array = [0,'EOB']
        else:
            for i in range(len(array)-1, -1, -1): # range(_, start, step)
                if array[i] != 0:
                    array.insert(i + 1, "EOB")
                    break

        precode = []
        zero_count = 0

        # Encode
        for i in range(len(array)):
            if i == 0: # DC
                precode.append((-1, array[i])) 
            elif array[i] == 'EOB':
                precode.append((-2, 'EOB'))
                break
            else:  
                if array[i] == 0:  # value == zero
                    zero_count = zero_count + 1
                    if zero_count > 15: # ZRL
                        precode.append((-3, 0))
                        zero_count = 0
                else:
                    precode.append((zero_count, array[i]))
                    zero_count = 0
    
        if self.print_once == 0:
            # Write_to_file("./Debug/precode_firstblock.txt", precode)
            self.print_once+=1
        
        self.precodes.append(precode)


    def Category(self, num):
        if num == 0:
            return 0
        
        abs_num = abs(num)
        exponent = 0
        
        while abs_num > 1:
            abs_num >>= 1  # 右移 除2
            exponent += 1
        
        return exponent + 1


    def LoadTable(self):
        with open("AC_Luminance.csv", newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                key = (row[0], row[1])
                value = row[2]
                self.ac_luminance_table[key] = value


    def ConvertInt2Binary(self, ans):
        tmp = abs(ans)
        binaryString = "{0:b}".format(tmp)

        if ans < 0 :
            tmp = ''
            for ch in binaryString:
                if ch=='0':
                    tmp+='1'
                else:
                    tmp+='0'
            return tmp
        else:
            return binaryString


    def Encode(self):
        '''
        Use self.precodes to encode
        '''
        pre_dc = 0

        for precode in self.precodes:
            for index, (value1, value2) in enumerate(precode):
                if value1 == -1: # DC
                    dc_diff = value2 - pre_dc
                    category_codeword = dc_table[self.Category(dc_diff)]
                    diff_codeword = self.ConvertInt2Binary(dc_diff)
                    codeword = category_codeword + diff_codeword
                    pre_dc = value2                   
                elif value1 == -2: # EOB
                    codeword = self.ac_luminance_table[('0', '0')]
                elif value1 == -3:
                    codeword = self.ac_luminance_table[('15', '0')]
                else:
                    size = self.Category(value2)
                    run_size_codeword = self.ac_luminance_table[(str(value1), str(size))]
                    ac_coefficient_codeword = self.ConvertInt2Binary(value2)
                    codeword = run_size_codeword + ac_coefficient_codeword
                
                self.encode_codeword += codeword


    def EncoderWriteFile(self):
        strstream = StringIO(self.encode_codeword)
        file = open('encoder_output', 'wb')
        while True:
            b = strstream.read(8)
            if not b:
                break
            if len(b) < 8:
                b = b + '0' * (8 - len(b))
            i = int(b, 2)
            file.write(i.to_bytes(1, byteorder='big'))
        
        file.close()


    def ScanZigzag(self):
        '''
        input blocks after DCT quantization
        called RunLengthCoding()
        '''

        for block in self.quantizedBlocks:
            array = [None] * (8*8) # one dimension array ---> len(array) = 64
            for i in range(8):
                for j in range(8):
                    array[zigZagMartixOrder[i,j]] = block[i,j]
            
            self.RunLengthCoding(array)

class DECODER:
    def __init__(self, qf) -> None:
        self.ac_luminance_table = {}
        self.qf = qf
        self.encoded_data_bit = ""
        self.precodes = []
        self.LoadTable()
        self.quantizedBlocks = [] # output after quantization


    def LoadTable(self):
        with open("AC_Luminance.csv", newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                key = row[2]
                value = (row[0], row[1])
                self.ac_luminance_table[key] = value


    def DecoderReadFile(self):
        with open('encoder_output', 'rb') as file:
            encoded_data_byte = file.read()

        self.encoded_data_bit = ''.join(format(byte, '08b') for byte in encoded_data_byte)


    def Find_in_dc_table(self, input_string):
        if input_string in dc_table:
            return True
        else:
            return False


    def InverseCategory(self, codeword): 
        if codeword[0] == '1': # positive number
            return int(codeword, 2)
        else:
            mask = '1' * len(codeword)
            return -(int(mask, 2) ^ int(codeword, 2))


    def Decode(self):
        strstream = StringIO(self.encoded_data_bit)

        dc_flag = True
        EOB_count = 0 
        pre_dc = 0
        buf = ""
        precode = []
        while True:
            buf += strstream.read(1)
            if dc_flag: 
                buf += strstream.read(1)
                isExist = self.Find_in_dc_table(buf)
                while not isExist:
                    buf += strstream.read(1)
                    isExist = self.Find_in_dc_table(buf)
                
                index = dc_table.index(buf)
                buf = ''
                if index == 0:
                    buf = strstream.read(1) # to read DIFF value codeword
                    num = 0
                else:
                    buf = strstream.read(index) # to read DIFF value codeword
                    num = self.InverseCategory(buf)  
                precode.append((-1, num + pre_dc))
                pre_dc = num + pre_dc
                dc_flag = False
                buf = ''
            elif buf == "1010":
                buf = ''
                precode.append((-2, 'EOB'))
                self.precodes.append(precode)
                precode = []
                dc_flag = True
                EOB_count += 1
                if EOB_count == 4096:
                    break
            elif buf == "11111111001":
                buf = ''
                precode.append((-3, 0))
            else:
                value = self.ac_luminance_table.get(buf)
                if value != None:
                    run = int(value[0])
                    size = int(value[1])
                    buf = ''
                    if size == 0:
                        num = 0
                    else:
                        buf = strstream.read(size) # to read AC coefficient codeword
                        num = self.InverseCategory(buf)
                    precode.append((run, num))
                    # print(f"precode {precode}")
                    buf = ''

        
    def InverseScanZigzag(self):
        deprecode_block = []
        deprecode_list = []
        for block in self.precodes:
            for precode in block:
                if precode[0] == -1: # DC
                    deprecode_block.append(precode[1])
                elif precode[0] == -2: # EOB
                    while len(deprecode_block) != 64:
                        deprecode_block.append(0)
                    deprecode_list.append(deprecode_block)
                    deprecode_block = []
                elif precode[0] == -3: # ZRL
                    for i in range(15):
                        deprecode_block.append(0)
                elif precode[0] == 0:
                    deprecode_block.append(precode[1])
                else: # EX: (1, 3)
                    for i in range(precode[0]):
                        deprecode_block.append(0)
                    deprecode_block.append(precode[1])

        
        # inverse zigzag
        for block in deprecode_list:
            # array = [None] * (8*8) # one dimension array ---> len(array) = 64
            dezigzag_block = [[0 for _ in range(8)] for _ in range(8)]
            for i in range(8):
                for j in range(8):
                    dezigzag_block[i][j] = block[zigZagMartixOrder[i,j]]
            
            self.quantizedBlocks.append(dezigzag_block)


    def PostprocessWriteFile(self, filename):
        if self.qf < 50:
            factor = 5000 / self.qf
        else:
            factor = 200 - (2 * self.qf)

        quantizationMatrix = np.floor(quantization_matrix_l * factor / 100).astype(np.int16)
        quantizationMatrix[quantizationMatrix==0] = 1

        deQuantizedBlocks = []
        for block in self.quantizedBlocks:
            deQuantizedBlocks.append(block*quantizationMatrix*1.)
        
        deDCTBlocks = []
        for block in deQuantizedBlocks:
            deDCTBlocks.append(np.round(cv2.idct(block)).astype(np.int8))
        
        img = np.ones((512,512),np.int8)*128

        img_pair = zip(deDCTBlocks, itertools.product(range(0,512,8),range(0,512,8)))
        for block , pair in img_pair:
            row_offset , col_offset = pair[0] , pair[1]
            img[row_offset:row_offset+8,col_offset:col_offset+8] += block

        filename = OUTPUT_PATH + filename
        cv2.imwrite(filename + ".jpg", img)
        return img


def Read_raw_image(filename, imgtype, width, height):
    with open(filename, 'rb') as file:
        raw_data = np.fromfile(file, dtype=np.uint8)
    if imgtype == "graylevel":
        image = raw_data.reshape((height, width))
    else:
        image = raw_data.reshape((height, width, 3))
    return image


def PSNR(file1, file2):
   mse = np.mean((file1 - file2) ** 2 )
   if mse < 1.0e-10:
      return 100
   return 10 * math.log10(255.0**2/mse)
 

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("qf", type=int)
    args = parser.parse_args()

    filename = args.path.split('\\')[-1].split('.')[0]
    out_filename = filename + str(args.qf)

    width = 512
    height = 512
    img = Read_raw_image(args.path, "graylevel", width, height)

    encoder = ENCODER(args.qf)
    encoder.Preprocess(img)
    encoder.ScanZigzag()
    encoder.Encode()
    encoder.EncoderWriteFile()

    decoder = DECODER(args.qf)
    decoder.DecoderReadFile()
    decoder.Decode()
    decoder.InverseScanZigzag()
    outimg = decoder.PostprocessWriteFile(out_filename)

    print(f"File: {out_filename}, PSNR: {PSNR(img, outimg)}")
