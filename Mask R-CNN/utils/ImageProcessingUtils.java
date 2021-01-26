/*
 * DeepImageJ
 * 
 * https://deepimagej.github.io/deepimagej/
 *
 * Conditions of use: You are free to use this software for research or educational purposes. 
 * In addition, we expect you to include adequate citations and acknowledgments whenever you 
 * present or publish results that are based on it.
 * 
 * Reference: DeepImageJ: A user-friendly plugin to run deep learning models in ImageJ
 * E. Gomez-de-Mariscal, C. Garcia-Lopez-de-Haro, L. Donati, M. Unser, A. Munoz-Barrutia, D. Sage. 
 * Submitted 2019.
 *
 * Bioengineering and Aerospace Engineering Department, Universidad Carlos III de Madrid, Spain
 * Biomedical Imaging Group, Ecole polytechnique federale de Lausanne (EPFL), Switzerland
 *
 * Corresponding authors: mamunozb@ing.uc3m.es, daniel.sage@epfl.ch
 *
 */

/*
 * Copyright 2019. Universidad Carlos III, Madrid, Spain and EPFL, Lausanne, Switzerland.
 * 
 * This file is part of DeepImageJ.
 * 
 * DeepImageJ is free software: you can redistribute it and/or modify it under the terms of 
 * the GNU General Public License as published by the Free Software Foundation, either 
 * version 3 of the License, or (at your option) any later version.
 * 
 * DeepImageJ is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
 * See the GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along with DeepImageJ. 
 * If not, see <http://www.gnu.org/licenses/>.
 */

package utils;

import ij.IJ;
import ij.ImagePlus;
import ij.process.ImageProcessor;

public class ImageProcessingUtils {
	
    /**
     * Resize image to wanted width and height
     * @param im: image to be resized
     * @param width: width to be resized
     * @param height: height to be resized
     * @return resized image
     */
    public static ImagePlus resize(ImagePlus im, int width, int height) {
    	im.getProcessor().setInterpolationMethod(2);
    	ImagePlus resizedImage = IJ.createHyperStack(im.getTitle(), width, height, im.getNChannels(), im.getNSlices(), im.getNFrames(), 32);
    	for (int c = 0; c < resizedImage.getNChannels(); c ++) {
    		for (int z = 0; z < resizedImage.getNFrames(); z ++) {
    			for (int t = 0; t < resizedImage.getNSlices(); t ++) {
    	    		im.setPositionWithoutUpdate(c + 1, z + 1, t + 1);
    	    		resizedImage.setPositionWithoutUpdate(c + 1, z + 1, t + 1);
    	    		ImageProcessor ip = im.getProcessor();
    	    		ImageProcessor op = ip.resize(width, height, true);
    	    		resizedImage.setProcessor(op);
    	    	}
        	}
    	}
    	return resizedImage;
    }
    
    /**
     * @param image: image to be padded
     * @param padding: number of values padded to the edges of each axis 
     * ((before_1,after_1), … (before_N, after_N))
     * @param value: value to which the padding will be set
     * @return padded image of the needed size
     */
    public static ImagePlus pad(ImagePlus image, double[][] padding, int value) {
    	int h = image.getHeight();
    	int w = image.getWidth();
    	int c = image.getNChannels();
    	int z = image.getNSlices();
    	int t = image.getNFrames();
    	int topPad = (int) padding[0][0];
    	int leftPad = (int) padding[1][0];
    	int newH = h + (int) padding[0][0] + (int) padding[0][1];
    	int newW = w + (int) padding[1][0] + (int) padding[1][1];
    	ImagePlus paddedIm = IJ.createHyperStack(image.getTitle(), newW, newH, c, z, t, 32);
    	ImageProcessor ipPad = null;
    	ImageProcessor ip;
    	for (int cc = 0; cc < c; cc ++) {
    		for (int tt = 0; tt < t; tt ++) {
    			for (int zz = 0; zz < z ; zz ++) {
					paddedIm.setPositionWithoutUpdate(cc + 1, zz + 1, tt + 1);
					image.setPositionWithoutUpdate(cc + 1, zz + 1, tt + 1);
    				for (int xx = 0; xx < w; xx ++) {
    					for (int yy = 0; yy < h; yy ++) {
    						ip = image.getProcessor();
    						ipPad = paddedIm.getProcessor();
    						ipPad.putPixelValue(xx + leftPad, yy + topPad, ip.getPixelValue(xx, yy));
    					}
    				}
    				paddedIm.setProcessor(ipPad);
    			}
    		}
    	}
    	return paddedIm;
    }

}
