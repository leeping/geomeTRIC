"""
info.py : Contains logo, citation messages, and other general information

The classes and subclasses in this module defines a "tree" relation of exceptions,
that can be used thoughout the code for a consistent error handling pattern.

Copyright 2016-2020 Regents of the University of California and the Authors

Authors: Lee-Ping Wang, Chenchen Song

Contributors:

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

import re

def colorString(word, color):
    if color == 'black':
        Answer = word
    elif color == 'blue':
        Answer = "\x1b[94m" + word +  "\x1b[0m"
    elif color == 'yellow':
        Answer = "\x1b[93m" + word +  "\x1b[0m"
    elif color == 'green':
        Answer = "\x1b[92m" + word +  "\x1b[0m"
    elif color == 'red':
        Answer = "\x1b[91m" + word +  "\x1b[0m"
    return Answer

def print_logo(logger=None):
    logostr="""
                                        ())))))))))))))))/                     
                                    ())))))))))))))))))))))))),                
                                *)))))))))))))))))))))))))))))))))             
                        #,    ()))))))))/                .)))))))))),          
                      #%%%%,  ())))))                        .))))))))*        
                      *%%%%%%,  ))              ..              ,))))))).      
                        *%%%%%%,         ***************/.        .)))))))     
                #%%/      (%%%%%%,    /*********************.       )))))))    
              .%%%%%%#      *%%%%%%,  *******/,     **********,      .))))))   
                .%%%%%%/      *%%%%%%,  **              ********      .))))))  
          ##      .%%%%%%/      (%%%%%%,                  ,******      /)))))  
        %%%%%%      .%%%%%%#      *%%%%%%,    ,/////.       ******      )))))) 
      #%      %%      .%%%%%%/      *%%%%%%,  ////////,      *****/     ,))))) 
    #%%  %%%  %%%#      .%%%%%%/      (%%%%%%,  ///////.     /*****      ))))).
  #%%%%.      %%%%%#      /%%%%%%*      #%%%%%%   /////)     ******      ))))),
    #%%%%##%  %%%#      .%%%%%%/      (%%%%%%,  ///////.     /*****      ))))).
      ##     %%%      .%%%%%%/      *%%%%%%,  ////////.      *****/     ,))))) 
        #%%%%#      /%%%%%%/      (%%%%%%      /)/)//       ******      )))))) 
          ##      .%%%%%%/      (%%%%%%,                  *******      ))))))  
                .%%%%%%/      *%%%%%%,  **.             /*******      .))))))  
              *%%%%%%/      (%%%%%%   ********/*..,*/*********       *))))))   
                #%%/      (%%%%%%,    *********************/        )))))))    
                        *%%%%%%,         ,**************/         ,))))))/     
                      (%%%%%%   ()                              ))))))))       
                      #%%%%,  ())))))                        ,)))))))),        
                        #,    ())))))))))                ,)))))))))).          
                                 ()))))))))))))))))))))))))))))))/             
                                    ())))))))))))))))))))))))).                
                                         ())))))))))))))),                     
"""

    b = 'blue'
    y = 'yellow'
    g = 'green'
    r = 'red'

    colorlist = [[],[r],[r],[r],[b,r,r],[b,r,r],[b,r,y,r],[b,y,r],[b,b,y,r],[b,b,y,y,r],
                 [b,b,y,y,r],[b,b,b,y,r],[b,b,b,g,y,r],[b,b,b,b,g,y,r],[b,b,b,b,b,g,y,r],
                 [b,b,b,b,g,y,r],[b,b,b,b,g,y,r],[b,b,b,b,g,y,r],[b,b,b,g,y,r],[b,b,b,y,r],
                 [b,b,y,y,r],[b,b,y,r],[b,b,y,r],[b,y,r],[b,r,r],[b,r,r],[b,r,r],[r],[r],[r]]
    
    words = [l.split() for l in logostr.split('\n')]
    for ln, line in enumerate(logostr.split('\n')):
        # Reconstruct the line.
        words = line.split()
        whites = re.findall('[ ]+',line)
        newline = ''
        i = 0
    
        if len(line) > 0 and line[0] == ' ':
            while i < max(len(words), len(whites)):
                try:
                    newline += whites[i]
                except: pass
                try:
                    newline += colorString(words[i], colorlist[ln][i])
                except: pass
                i += 1
        elif len(line) > 0:
            while i < max(len(words), len(whites)):
                try:
                    newline += colorString(words[i], colorlist[ln][i])
                except: pass
                try:
                    newline += whites[i]
                except: pass
                i += 1
        if logger is None:
            print(newline)
        else:
            logger.info(newline+'\n')

def print_citation(logger=None):
    logger.info("""
    #==========================================================================#
    #| If this code has benefited your research, please support us by citing: |#
    #|                                                                        |#
    #| Wang, L.-P.; Song, C.C. (2016) "Geometry optimization made simple with |#
    #| translation and rotation coordinates", J. Chem, Phys. 144, 214108.     |#
    #| http://dx.doi.org/10.1063/1.4952956                                    |#
    #==========================================================================#
    """)

