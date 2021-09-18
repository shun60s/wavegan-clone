#  wavegan clone   

This is cloned from  [chrisdonahue/wavegan](https://github.com/chrisdonahue/wavegan) to learn GAN synthesize.  


## experiment 1 

This is an expermental script to test original wavegan on google colaboratory.   
[wavegan_colab_practice1_sc09.ipynb](https://colab.research.google.com/github/shun60s/wavegan-clone/blob/master/wavegan_colab_practice1_sc09.ipynb)  

Following is 6 hours train result using sc09 (digits speech).  
D-loss G-loss (until 3k)  
![D-loss G-loss](waveGAN-SC09-D-Loss-G-Loss___until_3k.png)  
  
There are generated wave samples in model.ckp-2995-generated-wave-sample folder.  
Their sound quality is sill low, due to train time is short.  
  
  
## experiment 2  

This is a conditional wavegan, by adding condition to original wavegan using drum wave as label into wav 2nd channel.  

Following script is to test this conditional wavegan.   
[conditional_wavegan_colab_practice2_drum.ipynb](https://colab.research.google.com/github/shun60s/wavegan-clone/blob/master/conditional_wavegan_colab_practice2_drum.ipynb)  

There are generated drum samples in model.ckpt-4532-generated-drum-sample folder.  
This conditional result is still dissatisfaction. This condtion is probably not enough too sharp to discrimate.  


## License  

Regarding to original wavegan license, please see LICENSE-wavegan.txt.  
MIT  

