import numpy as np
from PIL import Image
import cv2
import argparse
import math
import dpcm

class NODE(object):
    def __init__(self, left=None, right=None):
        self.left = left
        self.right = right

    def children(self):
        return self.left, self.right

    def __str__(self):
        return self.left, self.right


class HUFFFMAN:
    def __init__(self):
        self.frequency = {}       
        self.frequency_decoder = []
        self.table = {}
        self.encode_output = ""
        self.decoded_array = np.zeros(65536, dtype=np.int16)
        

    def Count_frequency(self, img_array):
        for value in img_array:
            value_str = str(value)
            if value_str in self.frequency:
                self.frequency[value_str] += 1
            else:
                self.frequency[value_str] = 1

        self.frequency = sorted(self.frequency.items(), key=lambda x: x[1], reverse=True) # frequency from high to low
        with open('./Output/Huffman/frequency.txt', 'w') as f:
            for item in self.frequency:
                f.write(f'{item}\n')


    def Read_frequency_file(self, filename):
        with open(filename, 'r') as file:
            for line in file:
                key, value = line.strip('()\n').split(', ')
                key = key.strip("'")
                self.frequency_decoder.append((key, int(value)))


    def Build_tree(self, action):
        if action == "encode":
            nodes = self.frequency    
        else:
            nodes = self.frequency_decoder

        while len(nodes) > 1:
            (key1, frequency1) = nodes[-1]
            (key2, frequency2) = nodes[-2]
            nodes = nodes[:-2]
            node = NODE(key1, key2)
            nodes.append((node, frequency1 + frequency2))
            nodes = sorted(nodes, key=lambda x: x[1], reverse=True)
            # print(nodes)
        return nodes[0][0]
 

    def Encode_tree(self, node, binString=''):
        if type(node) is str:
            return {node: binString}
        (l, r) = node.children()
        self.table.update(self.Encode_tree(l, binString + '0'))
        self.table.update(self.Encode_tree(r, binString + '1'))
        return self.table


    def Encode_img(self, image_array, filename):
        '''Output encode to a txt file'''
        for pixel in image_array:
            pixel_str = str(pixel)
            if pixel_str in self.table:
                self.encode_output += self.table[pixel_str]
            else:
                assert("invalid symbol in huffman table")

        padbitnum = 8 - len(self.encode_output) % 8
        padbit = ''
        for i in range(padbitnum):
            padbit+=''
        self.encode_output = self.encode_output + padbit

        # use bytearray() to write file
        bytecode = bytearray()
        for i in range(0, len(self.encode_output), 8):
            byte = self.encode_output[i:i+8]
            bytecode.append(int(byte,2))

     
        path = (f"./Output/Huffman/{filename}_codewords.txt")
        with open(path, 'wb') as file:
            file.write(bytecode)


    def Decoder(self, isDPCM, isBinary,path="./Output/Huffman/frequency.txt"):
        self.Read_frequency_file(path)
        node = self.Build_tree("decode")
        current_node = node
        index = 0
        for bit in self.encode_output:
            if bit == '0':
                current_node = current_node.left
            else:
                current_node = current_node.right

            if type(current_node) is str:
                self.decoded_array[index] = int(current_node)
                index = index + 1
                current_node = node
        
        # print(f"self.decoded_array shape: {self.decoded_array}")
        if isDPCM:
            decode_output = dpcm.reverse_DPCM(self.decoded_array)
        else:
            decode_output = self.decoded_array
        if isBinary: 
            decode_output[decode_output == 1] = 255
        cv2.imwrite('./Output/Huffman/output.png', decode_output.reshape(256, 256))


    def Calculate_Performance(self):
        freq = {}
        entropy = 0.0
        totalsymbol = 65536
        totalLen = 0
        
        # Assuming self.frequency is a dictionary where keys are symbols and values are frequencies
        for symbol, frequency in self.frequency:
            probability = frequency / totalsymbol

            # Update entropy
            entropy -= probability * math.log2(probability)

            # update total len
            totalLen += len(self.table[symbol])*frequency 

        bits_avg = totalLen / totalsymbol
        print(f"=> Entropy: {entropy}")
        print(f"=> Bits average: {bits_avg}")
        print(f"=> Redundancy: {bits_avg-entropy}")
        
        return entropy


def Read_origin_img(path):
    image = Image.open(path)
    image_array = np.array(image)
    return image_array
    # img = cv2.imread(path,0)
    # return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("-dpcm", action="store_const", const=True, default=False)
    args = parser.parse_args()
    path = args.path
    isDPCM = args.dpcm

    huffman = HUFFFMAN()

    # split filename
    filename = args.path.split('/')[-1].split('.')[0]

    if "_b" or "_halftone" in filename:
        isBinary = True 
    else:
        isBinary = False
    
    if isDPCM:
        image_array = Read_origin_img(path)
        transform_image = dpcm.DPCM(image_array.flatten())
        huffman.Count_frequency(transform_image)
        node = huffman.Build_tree("encode")
        huffman.Encode_tree(node)
        encode_data = huffman.Encode_img(transform_image, filename)
        huffman.Decoder(isDPCM, isBinary)
        huffman.Calculate_Performance()

    else:
        image_array = Read_origin_img(path)
        huffman.Count_frequency(image_array.flatten())
        node = huffman.Build_tree("encode")
        huffman.Encode_tree(node)
        encode_data = huffman.Encode_img(image_array.flatten(), filename)
        huffman.Decoder(isDPCM, isBinary)
        huffman.Calculate_Performance()
           

