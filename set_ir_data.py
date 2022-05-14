# Based off of create_ir_segmentation.py from AirSim IR sensor page
import numpy
import cv2
import time
import sys
import os
import random
from airsim import *


def radiance(absoluteTemperature, emissivity, dx=0.01, response=None):
    """
    title::
        radiance

    description::
        Calculates radiance and integrated radiance over a bandpass of 8 to 14
        microns, given temperature and emissivity, using Planck's Law.

    inputs::
        absoluteTemperature
            temperture of object in [K]

            either a single temperature or a numpy
            array of temperatures, of shape (temperatures.shape[0], 1)
        emissivity
            average emissivity (number between 0 and 1 representing the
            efficiency with which it emits radiation; if 1, it is an ideal 
            blackbody) of object over the bandpass

            either a single emissivity or a numpy array of emissivities, of 
            shape (emissivities.shape[0], 1)
        dx
            discrete spacing between the wavelengths for evaluation of
            radiance and integration [default is 0.1]
        response
            optional response of the camera over the bandpass of 8 to 14 
            microns [default is None, for no response provided]
    
    returns::
        radiance
            discrete spectrum of radiance over bandpass
        integratedRadiance
            integration of radiance spectrum over bandpass (to simulate
            the readout from a sensor)

    author::
        Elizabeth Bondi
    """
    wavelength = numpy.arange(8,14,dx)
    c1 = 1.19104e8 # (2 * 6.62607*10^-34 [Js] * 
                   # (2.99792458 * 10^14 [micron/s])^2 * 10^12 to convert 
                   # denominator from microns^3 to microns * m^2)
    c2 = 1.43879e4 # (hc/k) [micron * K]
    if response is not None:
        radiance = response * emissivity * (c1 / ((wavelength**5) * \
                   (numpy.exp(c2 / (wavelength * absoluteTemperature )) - 1)))
    else:
        radiance = emissivity * (c1 / ((wavelength**5) * (numpy.exp(c2 / \
                   (wavelength * absoluteTemperature )) - 1)))
    if absoluteTemperature.ndim > 1:
        return radiance, numpy.trapz(radiance, dx=dx, axis=1)
    else:
        return radiance, numpy.trapz(radiance, dx=dx)


def get_new_temp_emiss_from_radiance(tempEmissivity, response):
    """
    title::
        get_new_temp_emiss_from_radiance

    description::
        Transform tempEmissivity from [objectName, temperature, emissivity]
        to [objectName, "radiance"] using radiance calculation above.

    input::
        tempEmissivity
            numpy array containing the temperature and emissivity of each
            object (e.g., each row has: [objectName, temperature, emissivity])
        response
            camera response (same input as radiance, set to None if lacking
            this information)

    returns::
        tempEmissivityNew
            tempEmissivity, now with [objectName, "radiance"]; note that 
            integrated radiance (L) is divided by the maximum and multiplied 
            by 255 in order to simulate an 8 bit digital count observed by the 
            thermal sensor, since radiance and digital count are linearly 
            related, so it's [objectName, simulated thermal digital count]

    author::
        Elizabeth Bondi
    """
    numObjects = tempEmissivity.shape[0]

    L = radiance(tempEmissivity[:,1].reshape((-1,1)).astype(numpy.float64), 
                 tempEmissivity[:,2].reshape((-1,1)).astype(numpy.float64), 
                 response=response)[1].flatten() 
    L = ((L / L.max()) * 255).astype(numpy.uint8)

    tempEmissivityNew = numpy.hstack((
        tempEmissivity[:,0].reshape((numObjects,1)), 
        L.reshape((numObjects,1))))

    return tempEmissivityNew

def set_segmentation_ids(segIdDict, tempEmissivityNew, client):
    """
    title::
        set_segmentation_ids

    description::
        Set stencil IDs in environment so that stencil IDs correspond to
        simulated thermal digital counts (e.g., if elephant has a simulated
        digital count of 219, set stencil ID to 219).

    input::
        segIdDict
            dictionary mapping environment object names to the object names in
            the first column of tempEmissivityNew 
        tempEmissivityNew
            numpy array containing object names and corresponding simulated
            thermal digital count
        client
            connection to AirSim (e.g., client = MultirotorClient() for UAV)

    author::
        Elizabeth Bondi
    """

    #Next set all objects of interest provided to corresponding object IDs
    #segIdDict values MUST match tempEmissivityNew labels.
    for key in segIdDict:
        objectID = int(tempEmissivityNew[numpy.where(tempEmissivityNew == \
                                                     segIdDict[key])[0],1][0])

        success = client.simSetSegmentationObjectID("[\w]*"+key+"[\w]*", 
                                                    objectID, True);
        if not success:
            print('There was a problem setting {0} segmentation object ID to {1!s}, or no {0} was found.'.format(key, objectID))
            
    time.sleep(0.1)

#if __name__ == '__main__':

def set_sensor():
    #Connect to AirSim, UAV mode.
    client = CarClient()
    client.confirmConnection()

    segIdDict = {
        'wall': 'wall',
        'car' : 'car',
        'hedge' : 'hedge',
        'landscape' : 'landscape',
        'road' : 'ground',
        'fir' : 'fir',
        'door' : 'door',
        'fence' : 'fence',
        'raccoon' : 'raccoon',
        'birch' : 'birch',
        'tree' : 'tree',
        'power_line' : 'power',
        'sign' : 'sign',
        'deer' : 'deer', 
        'foliage' : 'foliage',
        'leaves' : 'leaves',
        'house' : 'house',
        'driveway' : 'driveway',
        'roof' : 'roof',
        'pillar' : 'pillar',  
        'chimney' : 'chimney',
        'bench' : 'bench',
        'monument' : 'monument',
        'rock' : 'rock',
        'porch' : 'porch',
        'oak' : 'oak',
        'bin' : 'bin'
    }

    tempEmissivity = numpy.array([['wall', 295, 0.94],
                                  ['car', 280, 0.985],
                                  ['hedge', 288, 0.96],
                                  ['landscape', 283, 0.8],
                                  ['birch', 287, 0.90],
                                  ['fir', 286, 0.90],
                                  ['tree', 288, 0.90],
                                  ['door', 290, 0.91],
                                  ['raccoon', 300, 0.95],
                                  ['fence', 285, 0.90],
                                  ['power', 289, 0.90],
                                  ['sign', 288, 0.95],
                                  ['leaves', 279, 0.95],
                                  ['house', 287, 0.95],
                                  ['deer', 300, 0.95],
                                  ['foliage', 288, 0.95],
                                  ['driveway', 273, 0.95],
                                  ['roof', 310, 0.95],
                                  ['chimney', 278, 0.93],
                                  ['pillar', 286, 0.95],
                                  ['bench', 286, 0.95],
                                  ['bin', 281, 0.95],
                                  ['monument', 281, 0.95],
                                  ['rock', 290, 0.95],
                                  ['porch', 286, 0.95],
                                  ['oak', 285, 0.95],
                                  ['ground', 273, 0.958]])  

    response = None

    #Calculate radiance.
    tempEmissivityNew = get_new_temp_emiss_from_radiance(tempEmissivity, 
                                                         response)

    success = client.simSetSegmentationObjectID("[\w]*", 0, True);
    if not success:
        print('There was a problem setting all segmentation object IDs to 0. ')
        sys.exit(1)

    #Set IDs in AirSim environment.
    set_segmentation_ids(segIdDict, tempEmissivityNew, client)