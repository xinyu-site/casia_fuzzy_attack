__author__ = 'duguguang'

from nokov.nokovsdk import *
import time
import sys, getopt

preFrmNo = 0
curFrmNo = 0

def py_data_func(pFrameOfMocapData, pUserData):
    if pFrameOfMocapData == None:  
        print("Not get the data frame.\n")
    else:
        frameData = pFrameOfMocapData.contents
        global preFrmNo, curFrmNo 
        curFrmNo = frameData.iFrame
        if curFrmNo == preFrmNo:
            return

        preFrmNo = curFrmNo
        print( "FrameNo: %d\tTimeStamp:%Ld" % (frameData.iFrame, frameData.iTimeStamp))					
        print( "nMarkerset = %d" % frameData.nMarkerSets)

        for iMarkerSet in range(frameData.nMarkerSets):
            markerset = frameData.MocapData[iMarkerSet]
            print( "Markerset%d: %s [nMarkers Count=%d]\n" % (iMarkerSet+1, markerset.szName, markerset.nMarkers))
            print("{\n")

            for iMarker in range(markerset.nMarkers):
                print("\tMarker%d: %3.2f,%3.2f,%3.2f\n" %(	
				iMarker,
				markerset.Markers[iMarker][0],
				markerset.Markers[iMarker][1],
				markerset.Markers[iMarker][2]))
            print( "}\n")

        for iBody in range(frameData.nRigidBodies):
            body = frameData.RigidBodies[iBody]		
		
            # Markers
            print("\tRigidBody markers = %d\n" % body.nMarkers)
            print("\t{\n")
            for i in range(body.nMarkers):
                print("\t\tMarker %d：X:%f Y:%f Z:%f\n" % (
                i + 1,
                body.Markers[i][0],
                body.Markers[i][1],
                body.Markers[i][2]))

            print("\t}\n")  
		
        for iSkeleton in range(frameData.nSkeletons):
            # Segments
            skeleton = frameData.Skeletons[iSkeleton]
            print("\tnSegments = %d\n" % skeleton.nRigidBodies)
            print("\t{\n")
            for iBody in range(skeleton.nRigidBodies):
                body = skeleton.RigidBodyData[iBody]
                print("\t\tSegment %d：Tx:%lf Ty:%lf Tz:%lf qx:%lf qy:%lf qz:%lf qw:%lf\n" %
                    (iBody + 1,
                    body.x,
                    body.y,
                    body.z,
                    body.qx,
                    body.qy,
                    body.qz,
                    body.qw))

            print("\t}\n")
            print("}\n")

        # Unidentified Markers
        print("nUnidentifiedMarkers = %d" % frameData.nOtherMarkers)
        print("{\n")
        for i in range(frameData.nOtherMarkers):
            print("\tUnidentifiedMarkers %d：X:%f Y:%f Z:%f\n" %
                (i + 1,
                frameData.OtherMarkers[i][0],
                frameData.OtherMarkers[i][1],
                frameData.OtherMarkers[i][2]))
        
        print("}\n\n")

def py_msg_func(iLogLevel, szLogMessage):
    szLevel = "None"
    if iLogLevel == 4:
        szLevel = "Debug"
    elif iLogLevel == 3:
        szLevel = "Info"
    elif iLogLevel == 2:
        szLevel = "Warning"
    elif iLogLevel == 1:
        szLevel = "Error"
  
    print("[%s] %s" % (szLevel, cast(szLogMessage, c_char_p).value))

def py_forcePlate_func(pFocePlates, pUserData):
    if pFocePlates == None:  
        print("Not get the forcePlate frame.\n")
        pass
    else:
        ForcePlatesData = pFocePlates.contents
        print("iFrame:%d" % ForcePlatesData.iFrame)
        for iForcePlate in range(ForcePlatesData.nForcePlates):
            print("Fxyz:[%f,%f,%f] xyz:[%f,%f,%f] MFree:[%f]" % (
                ForcePlatesData.ForcePlates[iForcePlate].Fxyz[0],
                ForcePlatesData.ForcePlates[iForcePlate].Fxyz[1],
                ForcePlatesData.ForcePlates[iForcePlate].Fxyz[2],
                ForcePlatesData.ForcePlates[iForcePlate].xyz[0],
                ForcePlatesData.ForcePlates[iForcePlate].xyz[1],
                ForcePlatesData.ForcePlates[iForcePlate].xyz[2],
                ForcePlatesData.ForcePlates[iForcePlate].Mfree
            ))

def main(argv):
    serverIp = '10.1.1.198'

    try:
        opts, args = getopt.getopt(argv,"hs:",["server="])
    except getopt.GetoptError:
        print('SeekerSDKClient.py -s <serverIp>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('SeekerSDKClient.py -s <serverIp>')
            sys.exit()
        elif opt in ("-s", "--server"):
            serverIp = arg

    print ('serverIp is %s' % serverIp)
    print("Started the Seeker_SDK_Client Demo")
    client = PySDKClient()

    ver = client.PySeekerVersion()
    print('SeekerSDK Sample Client 2.4.0.3142(SeekerSDK ver. %d.%d.%d.%d)' % (ver[0], ver[1], ver[2], ver[3]))

    client.PySetVerbosityLevel(0)
    client.PySetMessageCallback(py_msg_func)
    client.PySetDataCallback(py_data_func, None)

    print("Begin to init the SDK Client")
    ret = client.Initialize(bytes(serverIp, encoding = "utf8"))

    if ret == 0:
        print("Connect to the Seeker Succeed")
    else:
        print("Connect Failed: [%d]" % ret)
        exit(0)

    #Give 5 seconds to system to init forceplate device
    ret = client.PyWaitForForcePlateInit(5000)
    if (ret != 0):
        print("Init ForcePlate Failed[%d]" % ret)
        exit(0)

    client.PySetForcePlateCallback(py_forcePlate_func, None)

    while(input("Press q to quit\n") != "q"):
        pass
 
if __name__ == "__main__":
   main(sys.argv[1:])