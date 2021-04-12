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

import ij.process.ImageProcessor;
import java.util.Set;
import ij.IJ;
import ij.measure.ResultsTable;
import ij.ImagePlus;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import deepimagej.processing.PostProcessingInterface;


public class PostProcessing implements PostProcessingInterface {
	/**
	 * Dictionary containing all the parameters parsed from the file
	 */
	private static HashMap<String, String> CONFIG = new HashMap<String, String>();
	/**
	 * Path to the other pre-processing file provided in deepImageJ. If it contains 
	 * either a .ijm or .txt file it will be parsed to find parameters
	 */
	private static String CONFIG_FILE_PATH;
	/**
	 * Attribute to communicate errors to DeepImageJ plugins
	 */
	private static String ERROR = "";

	/**
	 * Return error that stopped pre-processing to DeepImageJ
	 */
	@Override
	public String error() {
		return ERROR;
	}
	
	/**
	 * This method does the equivalent to unmold_detections at:
	 * https://github.com/matterport/Mask_RCNN/blob/3deaec5d902d16e1daf56b62d5971d428dc920bc/mrcnn/model.py#L2417
	 * 
	 * 
	 * Method containing the whole Java post-processing routine. 
	 * @param map: outputs to be post-processed. It is provided by deepImageJ. The keys
	 * correspond to name given by the model to the outputs. And the values are the images and 
	 * ResultsTables outputes by the model.
	 * @return this method has to return a HashMap with the post-processing results.
	 */
    public HashMap<String, Object> deepimagejPostprocessing(final HashMap<String, Object> map) {
        final Set<String> keys = map.keySet();
        ImagePlus mask = null;
        ResultsTable detections = null;
        for (final String k : keys) {
            if (k.equals(CONFIG.get("MRCNN_MASK"))) {
                mask = (ImagePlus) map.get(k);
            }
            else {
                if (!k.equals(CONFIG.get("MRCNN_DETECTION"))) {
                    continue;
                }
                detections = (ResultsTable) map.get(k);
            }
        }
        // Get the number of objects detected by the net
        final int nDetections = getNDetections(detections);
        // If nothing was detected just return null
        if (nDetections == 0) {
        	ERROR = "No object was detected in the input image.";
        	return null;
        }
        // Get the detected bounding boxes from the output table in normalised coodinates
        final double[][] boxes = new double[nDetections][4];
        // Get the class IDs of the detected objects
        final int[] classIds = new int[nDetections];
        for (int i = 0; i < nDetections; ++i) {
            boxes[i][0] = Double.parseDouble(detections.getStringValue(0, i));
            boxes[i][1] = Double.parseDouble(detections.getStringValue(1, i));
            boxes[i][2] = Double.parseDouble(detections.getStringValue(2, i));
            boxes[i][3] = Double.parseDouble(detections.getStringValue(3, i));
            classIds[i] = Integer.parseInt(detections.getStringValue(4, i));
        }
        // Select the masks corresponding to the objects detected
        final ImagePlus selectedMasks = IJ.createHyperStack("Processed " + mask.getTitle(), mask.getWidth(), mask.getHeight(), 1, nDetections, 1, 32);
        int z = 0;
        int[] array;
        for (int length = (array = classIds).length, l = 0; l < length; ++l) {
            final int classId = array[l];
            selectedMasks.setPositionWithoutUpdate(1, z + 1, 1);
            mask.setPositionWithoutUpdate(classId + 1, z + 1, 1);
            final ImageProcessor ip = mask.getProcessor();
            selectedMasks.setProcessor(ip);
            ++z;
        }

        // String get the needed parameters from the config file
        String originalShapeString = CONFIG.get("ORIGINAL_IMAGE_SIZE");
        String processingShapeString = CONFIG.get("PROCESSING_IMAGE_SIZE");
        String windowString = CONFIG.get("WINDOW_SIZE");
        // Get an float arrays from the strings
        float[] originalShape = str2array(originalShapeString);
        float[] processingShape = str2array(processingShapeString);
        float[] window = str2array(windowString);
        
        final ImagePlus finalMasks = IJ.createHyperStack("finalMask", (int) Math.floor(originalShape[1]), (int) Math.floor(originalShape[0]), 1, nDetections, 1, 32);
        // Denormalise the bounding boxes to pixel coordinates in the processing shape
        window = normBoxes(window, processingShape);
        float[] shift = {window[0], window[1], window[0], window[1]};
        // Window height
        float wh = window[2] - window[0];
        // Window width
        float ww = window[3] - window[1];
        float[] scale = {wh, ww, wh, ww};
        // Convert boxes to pixel coordinates of the original image
        for (int i = 0; i < boxes.length; i ++) {
        	for (int j = 0; j < boxes[0].length; j ++) {
        		boxes[i][j] = (boxes[i][j] - shift[j]) / scale[j];
        	}
        }
        
        // Set the interpolation method
        selectedMasks.getProcessor().setInterpolationMethod(2);
        // Get the final boxes that indicate where is the mask located in the image
        final int[][] scaledBoxes = denormBoxes(boxes, originalShape);
        // Paste the mask into their corresponding places
        for (int j = 0; j < classIds.length; ++j) {
            selectedMasks.setPositionWithoutUpdate(1, j + 1, 1);
            finalMasks.setPositionWithoutUpdate(1, j + 1, 1);
            ImageProcessor selectedMaskIp = selectedMasks.getProcessor();
            final ImageProcessor finalMaskIp = finalMasks.getProcessor();
            final int newHeight = scaledBoxes[j][2] - scaledBoxes[j][0];
            final int newWidth = scaledBoxes[j][3] - scaledBoxes[j][1];
            selectedMaskIp = selectedMaskIp.resize(newWidth, newHeight);
            int xSelected = -1;
            for (int xFinal = scaledBoxes[j][1]; xFinal < scaledBoxes[j][3]; ++xFinal) {
                ++xSelected;
                int ySelected = -1;
                for (int yFinal = scaledBoxes[j][0]; yFinal < scaledBoxes[j][2]; ++yFinal) {
                    ++ySelected;
                    final double val = selectedMaskIp.getPixelValue(xSelected, ySelected);
                    if (val >= 0.5) {
                        finalMaskIp.putPixelValue(xFinal, yFinal, 1.0);
                    }
                }
            }
        }
        mask.close();
        finalMasks.show();
        final HashMap<String, Object> outMap = new HashMap<String, Object>();
        outMap.put(finalMasks.getTitle(), finalMasks);
        outMap.put(detections.getTitle(), detections);
        return outMap;
    }

    /**
	 * Auxiliary method to be able to change some post-processing parameters without
	 * having to change the code. DeepImageJ gives the option of providing a extra
	 * files in the post-processing which can be used for example as config files.
	 * It can act as a config file because the needed parameters can be specified in
	 * a comment block and the parsed by the post-processing method
	 * @param configFiles: list of attachments. The files used by the post-processing
	 * can then be selected by the name 
	 */
    public void setConfigFiles(ArrayList<String> configFiles) {
	    	for (String ff : configFiles) {
	    		String fileName = ff.substring(ff.lastIndexOf(File.separator) + 1);
	    		if (fileName.contentEquals("config.ijm")) {
	    	    	CONFIG_FILE_PATH = ff;
	    	    	break;
	    		}
	    	}
	    	if (CONFIG_FILE_PATH == null && configFiles.size() == 0) {
	    		ERROR = "No parameters file or config file provided for post-processing.";
	    		return;
	    	} else if (CONFIG_FILE_PATH == null && configFiles.size() > 0) {
	    		ERROR = "A configuration file was not found in the model. The configuration file"
	    				+ "should be called 'config.ijm', please rename the config file if it is "
	    				+ "not named correctly.";
	    		return;
	    	} else if (!(new File(CONFIG_FILE_PATH).exists())) {
	    		ERROR = "The configuration file provided during post-processing does not exist.";
	    		return;
	    	}
	    	// Parse parameters from the config file
	    	// Parameters are saved in the HashMap 'config'
	    	getParameters(CONFIG_FILE_PATH);
	    }
    
    /**
     * Parse parameters from a file provided in the plugin.
     * This method will try to find if there is either a .ijm or .txt file provided for
     * post-processing and if there exists, it will be inputed to this method.
     * Inside this method now we can do whatever we want to the file to extract the wanted parameters.
     * In this particular case, the method will parse the text and look for the string: '* PARAMETER:' 
     * the corresponding parameter key will be right after, the '=' and finally the parameter value just before
     * the new line string '\n'
     * @param parametersFile: file containing parameters needed for post-processing provided in the plugin
     */
    public void getParameters(String parametersFile) {
    	// Initialise the parameters dictionary
    	CONFIG = new HashMap<String, String>();
    	File configFile = new File(parametersFile);
    	// Key that is used to know where is each parameter
    	String flag = "PARAMETER:";
    	String flag2 = "*";
    	String separator = "=";
    	// Read the file line by line
    	try (BufferedReader br = new BufferedReader(new FileReader(configFile))) {
    	    String line = br.readLine().trim();
    	    while (line != null) {
    	    	line = line.trim();
    	       if (line.contains(flag) && line.contains(flag2) && !line.contains("'" + flag + "'")) {
    	    	   int paramStart = line.indexOf(flag) + flag.length();
    	    	   int separatorInd = line.indexOf(separator);
    	    	   // Parameter key and value are separated by '='
    	    	   String key = line.substring(paramStart, separatorInd).trim();
    	    	   String value = line.substring(separatorInd + 1).trim();
    	    	   CONFIG.put(key, value);
    	       }
    	       line = br.readLine();
    	    }
    	    br.close();
    	} catch (IOException e) {
			ERROR = "Could not access the config file provided during pre-preocessing:\n"
					+ "- " + parametersFile;
			e.printStackTrace();
			CONFIG = null;
		}
    }
    
    /**
     * Get the number of objects detected by the model, that is rows that are non-zero
     * @param detections: ResultsTable with the output of the network
     * @return number of objects detected
     */
    private static int getNDetections(final ResultsTable detections) {
        int n = 0;
        for (int i = 0; i < detections.size(); ++i) {
            final int label = Integer.parseInt(detections.getStringValue(4, i));
            if (label != 0) {
                ++n;
            }
        }
        return n;
    }
    
    /**
     * Converts boxes from pixel coordinates to normalized coordinates.
     * Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
     * coordinates it's inside the box
     * @param window: array containing the coordinates of the window that corresponds to
     * the original image, [top, left, bottom, right], i.e: [ y1, x1, y2, x2 ]
     * @param imageShape: array with height and width of the modified image 
     * @return box normalized coordinates as [y1, x1, y2, x2]
     */
    private static float[] normBoxes(float[] window, float[] imageShape) {
    	float h = imageShape[0];
    	float w = imageShape[1];
    	float[] scale = {h - 1, w - 1, h - 1, w - 1}; 
    	float[] shift = {0, 0, 1, 1}; 
    	float[] normBox = new float[scale.length];
    	for (int i = 0; i < normBox.length; i ++)
    		normBox[i] = (window[i] - shift[i]) / scale[i];
    	return normBox;
    }
    
    /**
     * Converts boxes from normalized coordinates to pixel coordinates.
     * Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
     * coordinates it's inside the box.
     * @param boxes: vertices for the bounding boxes of each of the
     * detected objects. In normalised coordinates
     * @param shape: height and width in pixels of an image
     * @return an array containing the vertices of each bounding
     * box in pixel coordinates
     */
    private static int[][] denormBoxes(final double[][] boxes, final float[] shape) {
        final float h = shape[0];
        final float w = shape[1];
        final double[] scale = { h - 1, w - 1, h - 1, w - 1 };
        final double[] shift = { 0.0, 0.0, 1.0, 1.0 };
        final int[][] newBoxes = new int[boxes.length][boxes[0].length];
        for (int i = 0; i < boxes.length; ++i) {
            newBoxes[i][0] = (int)Math.round(boxes[i][0] * scale[0] + shift[0]);
            newBoxes[i][1] = (int)Math.round(boxes[i][1] * scale[1] + shift[1]);
            newBoxes[i][2] = (int)Math.round(boxes[i][2] * scale[2] + shift[2]);
            newBoxes[i][3] = (int)Math.round(boxes[i][3] * scale[3] + shift[3]);
        }
        return newBoxes;
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
}
