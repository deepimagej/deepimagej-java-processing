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

import ij.ImagePlus;

import java.util.HashMap;

import ij.IJ;

public class MaskRcnnAnchors {
	
    private static float[] RPN_ANCHOR_SCALES;
    private static float[] RPN_ANCHOR_RATIOS;
    private static float[] BACKBONE_STRIDES;
    private static float RPN_ANCHOR_STRIDE;
    
    
    public MaskRcnnAnchors(HashMap<String, String> config) {
    	RPN_ANCHOR_SCALES = str2array(config.get("RPN_ANCHOR_SCALES"));
        RPN_ANCHOR_RATIOS = str2array(config.get("RPN_ANCHOR_RATIOS"));
        BACKBONE_STRIDES = str2array(config.get("BACKBONE_STRIDES"));
        RPN_ANCHOR_STRIDE = Float.parseFloat(config.get("RPN_ANCHOR_STRIDE"));
    }
    
    /**
     * Converts an array of the form '[a,b,c,d]' into a float array
     * @param str: string representation of an array
     * @return float array or null in the case it was not possible
     */
    public static float[] str2array(String str) {
    	try {
	    	if (str.indexOf("[") != -1)
	    		str = str.substring(str.indexOf("[") + 1);
	    	else if (str.indexOf("(") != -1)
	    		str = str.substring(str.indexOf("(") + 1);
	
	    	if (str.indexOf("]") != -1)
	    		str = str.substring(0, str.indexOf("]"));
	    	else if (str.indexOf(")") != -1)
	    		str = str.substring(0, str.indexOf(")"));
	    	
	    	String[] strArr = str.split(",");
	    	float[] arr = new float[strArr.length];
	    	for (int i = 0; i < strArr.length; i ++) {
	    		arr[i] = Float.parseFloat(strArr[i]);
	    	}
	    	return arr;
    	} catch (Exception ex){
    		return null;
    	}
    }
    
    public static void main(final String[] args) {
        final ImagePlus im = IJ.createImage("aux", 1024, 1024, 1, 24);
        final float[][][] a = getAnchors(im);
    }
    
    public static float[][][] getAnchors(final ImagePlus im) {
        final float[] imShape = { (float)im.getHeight(), (float)im.getWidth(), (float)im.getNChannels() };
        final float[][] backboneShapes = computeBackboneShapes(imShape);
        final float[][] anchors = generatePyramidAnchors(MaskRcnnAnchors.RPN_ANCHOR_SCALES, MaskRcnnAnchors.RPN_ANCHOR_RATIOS, backboneShapes, MaskRcnnAnchors.BACKBONE_STRIDES, MaskRcnnAnchors.RPN_ANCHOR_STRIDE, imShape);
        final float[][][] tensorAnchors = new float[1][anchors.length][anchors[0].length];
        tensorAnchors[0] = anchors;
        return tensorAnchors;
    }
    
    public static float[][] computeBackboneShapes(final float[] imShape) {
        final float[][] backboneShapes = new float[MaskRcnnAnchors.BACKBONE_STRIDES.length][2];
        int c = 0;
        float[] backbone_STRIDES;
        for (int length = (backbone_STRIDES = MaskRcnnAnchors.BACKBONE_STRIDES).length, i = 0; i < length; ++i) {
            final float bs = backbone_STRIDES[i];
            backboneShapes[c][0] = (float)Math.ceil(imShape[0] / bs);
            backboneShapes[c++][1] = (float)Math.ceil(imShape[1] / bs);
        }
        return backboneShapes;
    }
    
    public static float[][] generatePyramidAnchors(final float[] scales, final float[] ratios, final float[][] featureShapes, final float[] featureStride, final float anchorStride, final float[] imshape) {
        int nAnchors = 0;
        final int nRatios = ratios.length;
        for (int i = 0; i < scales.length; ++i) {
            final int xStrides = (int)Math.floor(featureShapes[i][1] / anchorStride);
            final int yStrides = (int)Math.floor(featureShapes[i][0] / anchorStride);
            nAnchors += nRatios * xStrides * yStrides;
        }
        final float[][] anchors = new float[nAnchors][4];
        int ind = 0;
        for (int j = 0; j < scales.length; ++j) {
            final float[][] sAnchors = generateAnchors(scales[j], ratios, featureShapes[j], featureStride[j], anchorStride, imshape);
            System.arraycopy(sAnchors, 0, anchors, ind, sAnchors.length);
            ind += sAnchors.length;
        }
        return anchors;
    }
    
    public static float[][] generateAnchors(final float scale, final float[] ratios, final float[] shape, final float featureStride, final float anchorStride, final float[] imShape) {
        final float[] heights = new float[ratios.length];
        final float[] widths = new float[ratios.length];
        for (int i = 0; i < ratios.length; ++i) {
            heights[i] = (float)(scale / Math.sqrt(ratios[i]));
            widths[i] = (float)(scale * Math.sqrt(ratios[i]));
        }
        final float[] shiftsY = arange(0.0f, shape[0], anchorStride);
        for (int j = 0; j < shiftsY.length; ++j) {
            shiftsY[j] *= featureStride;
        }
        final float[] shiftsX = arange(0.0f, shape[1], anchorStride);
        for (int k = 0; k < shiftsX.length; ++k) {
            shiftsX[k] *= featureStride;
        }
        final float[] aux_x = new float[shiftsX.length * shiftsY.length];
        final float[] aux_y = new float[shiftsX.length * shiftsY.length];
        int count = 0;
        float[] array;
        for (int length = (array = shiftsY).length, n = 0; n < length; ++n) {
            final float y = array[n];
            float[] array2;
            for (int length2 = (array2 = shiftsX).length, n2 = 0; n2 < length2; ++n2) {
                final float x = array2[n2];
                aux_x[count] = x;
                aux_y[count++] = y;
            }
        }
        final float[][] boxWidthsMat = new float[aux_x.length][widths.length];
        final float[][] boxCentersXMat = new float[aux_x.length][widths.length];
        for (int w = 0; w < widths.length; ++w) {
            for (int x2 = 0; x2 < aux_x.length; ++x2) {
                boxWidthsMat[x2][w] = widths[w];
                boxCentersXMat[x2][w] = aux_x[x2];
            }
        }
        count = 0;
        final float[] boxWidths = new float[aux_x.length * widths.length];
        final float[] boxCentersX = new float[aux_x.length * widths.length];
        for (int x3 = 0; x3 < aux_x.length; ++x3) {
            for (int w2 = 0; w2 < widths.length; ++w2) {
                boxWidths[count] = boxWidthsMat[x3][w2];
                boxCentersX[count++] = boxCentersXMat[x3][w2];
            }
        }
        final float[][] boxHeightsMat = new float[aux_y.length][heights.length];
        final float[][] boxCentersYMat = new float[aux_y.length][heights.length];
        for (int h = 0; h < heights.length; ++h) {
            for (int y2 = 0; y2 < aux_y.length; ++y2) {
                boxHeightsMat[y2][h] = heights[h];
                boxCentersYMat[y2][h] = aux_y[y2];
            }
        }
        count = 0;
        final float[] boxHeights = new float[aux_y.length * heights.length];
        final float[] boxCentersY = new float[aux_y.length * heights.length];
        for (int y3 = 0; y3 < aux_y.length; ++y3) {
            for (int h2 = 0; h2 < heights.length; ++h2) {
                boxHeights[count] = boxHeightsMat[y3][h2];
                boxCentersY[count++] = boxCentersYMat[y3][h2];
            }
        }
        final float scaleY = imShape[0] - 1.0f;
        final float scaleX = imShape[1] - 1.0f;
        final float shift = 1.0f;
        final float[][] boxes = new float[boxCentersY.length][4];
        for (int l = 0; l < boxes.length; ++l) {
            boxes[l][0] = (float)(boxCentersY[l] - boxHeights[l] * 0.5) / scaleY;
            boxes[l][1] = (float)(boxCentersX[l] - boxWidths[l] * 0.5) / scaleX;
            boxes[l][2] = ((float)(boxCentersY[l] + boxHeights[l] * 0.5) - shift) / scaleY;
            boxes[l][3] = ((float)(boxCentersX[l] + boxWidths[l] * 0.5) - shift) / scaleX;
        }
        return boxes;
    }
    
    private static float[] arange(float start, final float end, final float space) {
        final int nComponents = (int)Math.floor((end - start) / (double)space);
        final float[] arr = new float[nComponents];
        for (int i = 0; start < end; start += space, ++i) {
            arr[i] = start;
        }
        return arr;
    }
}
