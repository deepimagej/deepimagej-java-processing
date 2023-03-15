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
package maskrcnn.utils;

import io.bioimage.modelrunner.utils.IndexingUtils;
import net.imglib2.Cursor;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.type.numeric.real.FloatType;

public class ImgLib2Builder {

    
    public static Img<FloatType> createTensorFromArray(float[] flatArr, long[] tensorShape){
		final ArrayImgFactory<FloatType> factory = new ArrayImgFactory<>(new FloatType());
		final Img<FloatType> outputImg = factory.create(tensorShape);
		Cursor<FloatType> tensorCursor = outputImg.cursor();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
			int flatPos = IndexingUtils.multidimensionalIntoFlatIndex(cursorPos,
				tensorShape);
			float val = flatArr[flatPos];
			tensorCursor.get().set(val);
		}
		return outputImg;
    }
    
    
    public static Img<FloatType> createTensorFromArray(float[][] flatArr){
    	long[] tensorShape = new long[] {flatArr.length, flatArr[0].length};
		final ArrayImgFactory<FloatType> factory = new ArrayImgFactory<>(new FloatType());
		final Img<FloatType> outputImg = factory.create(tensorShape);
		Cursor<FloatType> tensorCursor = outputImg.cursor();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
			float val = flatArr[(int) cursorPos[0]][(int) cursorPos[1]];
			tensorCursor.get().set(val);
		}
		return outputImg;
    }
    
    
    public static Img<FloatType> createTensorFromArray(float[][][] flatArr){
    	long[] tensorShape = new long[] {flatArr.length, flatArr[0].length, flatArr[0][0].length};
		final ArrayImgFactory<FloatType> factory = new ArrayImgFactory<>(new FloatType());
		final Img<FloatType> outputImg = factory.create(tensorShape);
		Cursor<FloatType> tensorCursor = outputImg.cursor();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
			float val = flatArr[(int) cursorPos[0]][(int) cursorPos[1]][(int) cursorPos[2]];
			tensorCursor.get().set(val);
		}
		return outputImg;
    }
    
    
    public static Img<FloatType> createTensorFromArray(float[][][][] flatArr){
    	long[] tensorShape = new long[] {flatArr.length, flatArr[0].length, 
    			flatArr[0][0].length, flatArr[0][0][0].length};
		final ArrayImgFactory<FloatType> factory = new ArrayImgFactory<>(new FloatType());
		final Img<FloatType> outputImg = factory.create(tensorShape);
		Cursor<FloatType> tensorCursor = outputImg.cursor();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
			float val = flatArr[(int) cursorPos[0]][(int) cursorPos[1]][(int) cursorPos[2]][(int) cursorPos[3]];
			tensorCursor.get().set(val);
		}
		return outputImg;
    }
    
    
    public static Img<FloatType> createTensorFromArray(float[][][][][] flatArr){
    	long[] tensorShape = new long[] {flatArr.length, flatArr[0].length, 
    			flatArr[0][0].length, flatArr[0][0][0].length, flatArr[0][0][0][0].length};
		final ArrayImgFactory<FloatType> factory = new ArrayImgFactory<>(new FloatType());
		final Img<FloatType> outputImg = factory.create(tensorShape);
		Cursor<FloatType> tensorCursor = outputImg.cursor();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
			float val = 
					flatArr[(int) cursorPos[0]][(int) cursorPos[1]][(int) cursorPos[2]][(int) cursorPos[3]][(int) cursorPos[4]];
			tensorCursor.get().set(val);
		}
		return outputImg;
    }
}
