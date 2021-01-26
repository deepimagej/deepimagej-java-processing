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
import ij.IJ;

public class MaskRcnnMetas {
	
    private double IMAGE_MIN_DIM;
    private double IMAGE_MIN_SCALE;
    private double IMAGE_MAX_DIM;
    private String IMAGE_RESIZE_MODE;
    private double NUM_CLASSES;
    private static double INPUT_SCALE;
    private static double[] INPUT_WINDOW;
    private static double[][] INPUT_PADDING;
    private static double[] MEAN_PIXEL;
    private static float[] RPN_ANCHOR_SCALES;
    private static float[] RPN_ANCHOR_RATIOS;
    private static float[] BACKBONE_STRIDES;
    private static float RPN_ANCHOR_STRIDE;
    private static float id;
    private static int nClasses;
    private static float scale;
    
    static {
        MaskRcnnMetas.INPUT_WINDOW = new double[4];
        MaskRcnnMetas.INPUT_PADDING = new double[4][2];
        MaskRcnnMetas.MEAN_PIXEL = new double[] { 123.7, 116.8, 103.9 };
        MaskRcnnMetas.RPN_ANCHOR_SCALES = new float[] { 32.0f, 64.0f, 128.0f, 256.0f, 512.0f };
        MaskRcnnMetas.RPN_ANCHOR_RATIOS = new float[] { 0.5f, 1.0f, 2.0f };
        MaskRcnnMetas.BACKBONE_STRIDES = new float[] { 4.0f, 8.0f, 16.0f, 32.0f, 64.0f };
        MaskRcnnMetas.RPN_ANCHOR_STRIDE = 1.0f;
        MaskRcnnMetas.id = 0.0f;
        MaskRcnnMetas.nClasses = 81;
        MaskRcnnMetas.scale = 1.0f;
    }
    
    public MaskRcnnMetas() {
        this.IMAGE_MIN_DIM = 800.0;
        this.IMAGE_MIN_SCALE = 0.0;
        this.IMAGE_MAX_DIM = 1024.0;
        this.IMAGE_RESIZE_MODE = "square";
        this.NUM_CLASSES = 81.0;
    }
    
    public static void main(final String[] args) {
        final ImagePlus im = IJ.createImage("aux", 1024, 1024, 3, 24);
        final float[][] a = composeImageMeta(im);
        a[0][0] = 2.0f;
    }
    
    public static float[][] composeImageMeta(final ImagePlus im) {
        final float[] originalImShape = { (float)im.getHeight(), (float)im.getWidth(), (float)im.getNChannels() };
        final float[] finalShape = { (float)im.getHeight(), (float)im.getWidth(), (float)im.getNChannels() };
        final float[] window = { 0.0f, 0.0f, finalShape[0], finalShape[1] };
        final float[][] imageMetas = composeImageMeta(MaskRcnnMetas.id, originalImShape, finalShape, window, MaskRcnnMetas.scale, MaskRcnnMetas.nClasses);
        return imageMetas;
    }
    
    public static float[][] composeImageMeta(final float id, final float[] originalImShape, final float[] finalShape, final float[] window, final float scale, final int nClasses) {
        final float[] classesArray = new float[nClasses];
        final int metaSize = 1 + originalImShape.length + finalShape.length + window.length + 1 + nClasses;
        final float[] meta = new float[metaSize];
        int i = 0;
        meta[i++] = id;
        for (int c = 0; c < originalImShape.length; ++c) {
            meta[i++] = originalImShape[c];
        }
        for (int c = 0; c < finalShape.length; ++c) {
            meta[i++] = finalShape[c];
        }
        for (int c = 0; c < window.length; ++c) {
            meta[i++] = window[c];
        }
        meta[i++] = scale;
        for (int c = 0; c < classesArray.length; ++c) {
            meta[i++] = classesArray[c];
        }
        final float[][] metaTensor = new float[1][meta.length];
        metaTensor[0] = meta;
        return metaTensor;
    }
}
