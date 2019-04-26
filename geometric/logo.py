#!/usr/bin/env python

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

def printLogo(logger=None):
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
