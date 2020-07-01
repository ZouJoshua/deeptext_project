import os
import shutil
#os.remove('Best.txt')                                                                                         

shutil.rmtree('export_model', ignore_errors=True)                                                                                                                                                  
#shutil.rmtree('tfrecord', ignore_errors=True)                                                                                                                                                      
shutil.rmtree('eval_dir', ignore_errors=True)                                               
shutil.rmtree('checkpoint', ignore_errors=True)                                                                                                                                
shutil.rmtree('__pycache__', ignore_errors=True)
os.remove('Best.txt')
