#include <stdio.h>
#include "errhand.h"
#include "bitio.h"

void CompressFile_qm(BIT_FILE *input,BIT_FILE *output,long input_file_size);
void ExpandFile_qm(BIT_FILE *input,BIT_FILE *output);
void renorm_e(BIT_FILE *output);
void renorm_d(BIT_FILE *input);
void byte_out(BIT_FILE *output);
void byte_in(BIT_FILE *input);
void flush(BIT_FILE *output);

/***** State Table *****/
int next_state_lps[46] = {   0,     1,     1,     1,     1,     1,     1,     1,     2,     1,     2,     1,     1,     2,     1,     2,     1,     2,     2,     1,     2,     2,     2,     2,     1,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     1,     2,     2,     2,     2,     2,     3,     2,     2,     2,     1};
unsigned long Qe[46] = { 0x59eb,0x5522,0x504f,0x4b85,0x4639,0x415e,0x3c3d,0x375e,0x32b4,0x2e17,0x299a,0x2516,0x1edf,0x1aa9,0x174e,0x1424,0x119c,0x0f6b,0x0d51,0x0bb6,0x0a40,0x0861,0x0706,0x05cd,0x04de,0x040f,0x0363,0x02d4,0x025c,0x01f8,0x01a4,0x0160,0x0125,0x00f6,0x00cb,0x00ab,0x008f,0x0068,0x004e,0x003b,0x002c,0x001a,0x000d,0x0006,0x0003,0x0001};
unsigned long C;     /* 32 bits */
unsigned long A;
unsigned long SC = 0;
unsigned char code_buffer;
unsigned char is_data_in_code_buffer = 0;
unsigned char CT;
unsigned char is_terminate = 0;
/****************************************************/
void CompressFile_qm(BIT_FILE *input,BIT_FILE *output,long input_file_size)
{
  int new_bit;   /* declared "int" in order to intercept EOF */  
  int state = 0;
  int current_mps = 0;
  int state_change;
  int stuffing = 0;
  C = 0;
  A = 0x10000;
  CT = 11;
  
  /* Output Total character number */
  OutputBits( output, input_file_size, 32);
    
  /* Starting QM encoding */
  do
  {
      new_bit = InputBit(input);
      if(new_bit == EOF) { new_bit = 0; stuffing++; }
      
      A -= Qe[state];
      
      if( new_bit == current_mps )
      {
          if(A<0x8000)
          {
              if(A<Qe[state]) { C+=A;A=Qe[state]; }
              if(state<45) state++; /* Estimate Qe after MPS */
              renorm_e(output);
          }
      }
      else
      {
          if(A>=Qe[state]) { C+=A; A=Qe[state]; }
          if(state==0) current_mps = 1 - current_mps;
          state_change = next_state_lps[state];
          state -= state_change;
          renorm_e(output);
      }
  }  while( stuffing < 20 );

  flush(output);
}

/*****************************************************************/

void ExpandFile_qm(BIT_FILE *input,BIT_FILE *output)
{
  long output_file_bits;
  int output_bit; 
  int state = 0;
  int current_mps = 0;
  
  output_file_bits = 8 * ((long) InputBits(input,32));
  C = InputBits(input,16);
  C <<= 16;
  A = 0;
  CT = 0;

 
  /* Starting QM encoding */
  for(;output_file_bits>0; output_file_bits--) 
  {
      A -= (Qe[state] << 16);
      output_bit = current_mps;
      
      if(C<A)
      {
          if(A<0x80000000)
          {
              if(A < (Qe[state] << 16) )
              {
                  output_bit = 1 - current_mps;
                  if(state==0) current_mps = 1 - current_mps;
                  state -= next_state_lps[state];                  
              }
              else if(state<45) state++;
              renorm_d(input);
          }
      }
      else
      { 
          if(A >= (Qe[state] << 16) )
          {
              C -= A;
              A = Qe[state] << 16;
              output_bit = 1 - current_mps;     
              if(state==0) current_mps = 1 - current_mps;
              state -= next_state_lps[state];
          }
          else
          {
              C -= A;
              A = Qe[state] << 16;  
              if(state<45) state++;
          }
          renorm_d(input);
      }
      OutputBit(output,output_bit);
  }
  
}

/*********************************************************/

void renorm_e(BIT_FILE *output)
{
   do
   {
     A <<= 1;       
     C <<= 1;
     CT--;
     if(CT==0) { byte_out(output); CT=8; }
   } while(A<0x8000);
}


void renorm_d(BIT_FILE *input)
{
   do
   {
     if(CT==0) { byte_in(input); CT=8; }
     A <<= 1;       
     C <<= 1;
     CT--;
   } while(A<0x80000000);
}

/*********************************************************************/
void byte_out(BIT_FILE *output)
{                            /* Never put 0xff in code_buffer */
   unsigned long T;
   
   T = (C >> 19);

   if(T>0xff)
   {
       code_buffer += 1;
       if(is_data_in_code_buffer) OutputBits(output,code_buffer,8);
       /*if(code_buffer == 0xff) fputc(0,output);*/
       for(;SC>0;SC--) OutputBits(output,0,8);
       if(SC != 0 ) printf("Error: SC != 0\n");
       code_buffer = (unsigned char) (T & 0x000000ff);
       /*printf("Carry produced! \n");*/
   }
   if(T<0xff)   
   {
       if(is_data_in_code_buffer) OutputBits(output,code_buffer,8); 
       for(;SC>0;SC--) { OutputBits(output,0xff,8); /*fputc(0,output);*/ }
       if(SC != 0 ) printf("Error: SC != 0\n");
       code_buffer = (unsigned char) (T & 0x000000ff);
   }
   if(T==0xff) SC += 1;

   C &= 0x0007ffff;
   is_data_in_code_buffer = 1;  
}


void byte_in(BIT_FILE *input)
{
   int input_buffer;
 
 /*  if(is_terminate) return;
     
   if((input_buffer = InputBits(input,8))==0xff)
   {
       if((input_buffer = InputBits(input,8)) == 0)
          C |= 0x0000ff00;
       else 
       {
           C &= 0xffff0000;
           is_terminate = 1;
       }   
   }
   else C += (input_buffer << 8);
 */
   input_buffer = InputBits(input,8);
   if(input_buffer==EOF) input_buffer = 0; 
   C += (input_buffer << 8);   
   CT = 8;
}



void flush(BIT_FILE *output)
{
   unsigned long T;

   T = (C+A-1) & 0xffff0000;
   if(T<C) T += 0x8000;
   C = (T << CT);
   byte_out(output);
   C <<= 8;
   byte_out(output);
}
