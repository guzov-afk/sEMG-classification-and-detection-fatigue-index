from   PyQt5.QtWidgets  import  *
from   PyQt5.QtGui      import  *
from   PyQt5.QtCore     import  QTimer, QEventLoop, QThread, QThreadPool, QMutex, pyqtSignal, pyqtSlot  # Components necessary for Threading
from   pyqtgraph        import  PlotWidget, plot                                    # GUI Libraries
from   pyqtgraph.Qt     import  QtGui, QtCore      

#import tensorflow       as      tf
import pyqtgraph        as      pg                                                  # plotting and
import numpy            as      np                                                  # Number manipulation

import socket                                                                       # UDP manipulation
import serial                                                                       # Serial Communication
import time                                                                         # for measuring purposes
import glob                                                                         # OS related libraries
import sys
import os

baud_rate               =       9600
COM_port                =       "COM11"

packet_size             =       1024                                                # dimension (in samples) of the wireless data packet
start_cmd               =       "START".encode("utf-8")                             # Start command given to the Armband to sample
retry_cmd               =       "RETRANSMIT".encode("utf-8")                             # Start command given to the Armband to sample

UDP_IP                  =       "192.168.4.1"                                       # IP address of the ESP32 SoftAP
UDP_PORT                =       1234                                                # Port used for UDP transmission
sock                    =       None

channel_number          =       8                                                   # channel number by default
plot_time_length        =       1                                                   # the time portion to be displayed, in seconds
sample_rate             =       512                                                 # the default sampling rate (per channel) of the device
window_size             =       int(plot_time_length * sample_rate)                 # how many samples will be displayed
frame_rate              =       4                                                   # how many times per second is the plot refreshed
frame_time              =       int( 1000 / frame_rate )                            # frame display duration, in miliseconds
plot_column_count       =       1                                                   # the number of plots per row
sliding_window          =       int(packet_size / channel_number)                   # the portion of the graph that is constantly updating
offset_val              =       128                                                 # Middle value of the graph


#-----------------------------------------------------------------------------
# Armband and Servo-Circuit Connection
'''
servo_con = serial.Serial( port = COM_port,
                           baudrate = baud_rate,
                           bytesize = 8,
                           timeout = 1,
                           stopbits = serial.STOPBITS_ONE)

servo_con.set_buffer_size( rx_size = 100, tx_size = 100 )
'''

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)         
sock.connect((UDP_IP, UDP_PORT))                            # client PC uses this
sock.settimeout(0.002)                                        # set a timeout in case the receive function timeouts
sock.send(start_cmd)


#-----------------------------------------------------------------------------
# Classes and Subroutine section
#-----------------------------------------------------------------------------

class Window(QMainWindow):

    #-----------------------------------------------------------------------------
    # Parameter Section - Configurable by End User


    acquisition_time        =   1                                                   # Acquisition time, in seconds 
         
    window_width            =   1280                                                # window dimensions in pixels
    window_height           =   720

    # Plot Parameters:
    # Packets of 1000 Sa/s
    # 125 Sa for each of the 8 channels in a packet

    plot_step               =   0
    traces                  =   []
    channel_buffers         =   offset_val * np.ones((channel_number, window_size), dtype=np.uint8)
    start_moment = time.time_ns()

    # File Recording variables and PyQt Signals
    record_name             =   None
    record_duration         =   5
    record_status           =   False
    packets_to_record       =   0

    connection_status       =   False
    acquire_status          =   False

    record_on_sgn           =   pyqtSignal(bool)
    record_counter_sgn      =   pyqtSignal(int)
    record_name_sgn         =   pyqtSignal(str)
    record_payload_sgn      =   pyqtSignal(list)

    ml_payload_sgn          =   pyqtSignal(list)
       
    def __init__(self):
        super().__init__()
        self.mutex = QMutex()
 
        self.setWindowTitle("sEMG Acquisition V1") 
        self.setGeometry(100, 100, self.window_width, self.window_height)       # setting geometry :offset horizontal, offset vertical, size horizontal, size vertical
        self.UiComponents()                                                     # calling method
        self.new_samples = None

        #Acquisition Thread Initialization
        
        self.retrieve_thread = portRead()
        self.retrieve_thread.sgn.connect(self.update)
        self.retrieve_thread.start()

        #Recorder Thread Initialization

        self.storage_thread = dataRecord()
        self.record_name_sgn.connect(self.storage_thread.set_filename)          # first 4 signals send data to the thread
        self.record_on_sgn.connect(self.storage_thread.toggle_on_off)
        self.record_counter_sgn.connect(self.storage_thread.set_packet_count)
        self.record_payload_sgn.connect(self.storage_thread.recorder)
        self.storage_thread.done_sgn.connect(self.record_done)                  # last one receives confirmatio that everything is ok
        self.storage_thread.start()

        #Neural Network Thread Initialization

        #self.classifier = ML_classifier()
        #self.classifier.doneSignal.connect(self.gesture_result)
        #self.ml_payload_sgn.connect(self.classifier.analyze)
        #self.classifier.start()

        self.show()                                                             # showing all the widgets

    def UiComponents(self):                         # method for components

        widget = QWidget()                          # creating a widget object For the Main Window
        # creating a plot window
        win = pg.plot()
        win.showGrid(x = True, y = True)
        win.setXRange(0, plot_time_length * sample_rate, padding = False)
        win.setYRange(0, 257 * channel_number, padding = False)
        win.setMouseEnabled(x = False, y = False)        

        colors      =   [ (255,  0,  0),       # red                                    # every plot with its own color for easier visualisation
                          (255,160,  0),       # orange
                          (255,255,  0),       # yellow
                          (0  ,255,  0),
                          (255,128,255),
                          (255,0  ,255),
                          (0  ,255,255),
                          ( 80,255,  0)]       # green
                             

        for ch in range(channel_number):                    # set the plots according to the number of channels
            trace = pg.PlotCurveItem(pen=({'color': colors[ch], 'width': 1}),name = 'Ch '+ str(ch + 1), skipFiniteCheck=True)
            win.addItem(trace)                              # each plot with it's own color
            trace.setPos(0, ch * 256)                       # offset it on the chart with one channel size
            self.traces.append(trace)                                
            trace.setData(self.channel_buffers[ch])

        ### USER Input Layout and Widgets
        user_input_layout            = QVBoxLayout()        
        user_inputs                  = QWidget()
        
        self.label_sample_rate       = QLabel('Armband Sample Rate: ' + str(sample_rate) + "Sa/s - per ch", self)
        self.label_channel_number    = QLabel('Channel Count: ' + str(channel_number))
        self.label_connected         = QLabel('Armband Status: Disconnected')
    

        self.label_sample_rate.setFont(QFont('Arial',12))
        self.label_channel_number.setFont(QFont('Arial',12))
        self.label_connected.setFont(QFont('Arial',12))
        

        # widget for displaying the gesture performed

        self.label_command  = QLabel("Command: ", self)
        self.label_command.setFont(QFont('Arial',12))

        self.label_arrow = QLabel(self)
        self.label_arrow.setScaledContents(True)
        
        #self.gesture_0_sign = QPixmap('0.png')
        #self.gesture_1_sign = QPixmap('1.png')
        #self.gesture_2_sign = QPixmap('2.png')
        #self.gesture_3_sign = QPixmap('3.png')
        #self.gesture_4_sign = QPixmap('4.png')
  
        #self.label_arrow.setPixmap(self.gesture_0_sign)

        # Displaying information regarding the Armband connection

        self.label_IP                = QLineEdit('Armband IP: ' + UDP_IP, self)
        self.label_IP.setFont(QFont('Arial', 12))
        self.label_IP.textChanged[str].connect(self.change_IP)
        
        self.label_PORT              = QLineEdit('Armband PORT: ' + str(UDP_PORT), self)
        self.label_PORT.setFont(QFont('Arial', 12))
        self.label_PORT.textChanged[str].connect(self.change_PORT)
        
        
        self.connect_button          = QPushButton('Connect Armband', self)
        self.connect_button.setCheckable(True)
        self.connect_button.setFont(QFont('Arial', 12))
        self.connect_button.clicked.connect(self.connect_armband)
        
        self.acquire_button          = QPushButton('Start Acquisition', self)
        self.acquire_button.setCheckable(True)
        self.acquire_button.setFont(QFont('Arial', 12))
        self.acquire_button.clicked.connect(self.start_stop_acquisition)
        
        
        self.record_duration         = QLineEdit("3")                               # creating a line edit widget
        self.record_duration.setFont(QFont('Arial', 12))
        self.record_duration.textChanged[str].connect(self.set_record_duration)
        
        self.record_name             = QLineEdit("File Name")                       # creating a line edit widget
        self.record_name.setFont(QFont('Arial', 12))
        self.record_name.textChanged[str].connect(self.file_rename)
        #self.check                   = QCheckBox("Check Box")                      # creating a check box widget
        
        self.record_button           = QPushButton('Start Recording Data', self)    # creating a push button object
        self.record_button.setFont(QFont('Arial', 12))
        self.record_button.setCheckable(True)
        self.record_button.clicked.connect(self.record)
        self.record_button.setToolTip('Start/Stop Recording')
        
    
        # adding the necessary widgets
        user_input_layout.addWidget(self.label_command)
        user_input_layout.addWidget(self.label_arrow)
        user_input_layout.addWidget(self.label_sample_rate)
        user_input_layout.addWidget(self.label_channel_number)

        user_input_layout.addWidget(self.label_connected)
        user_input_layout.addWidget(self.label_IP)
        user_input_layout.addWidget(self.label_PORT)
      
        user_input_layout.addWidget(self.connect_button)
        user_input_layout.addWidget(self.acquire_button)
        
        user_input_layout.addWidget(self.record_name)
        user_input_layout.addWidget(self.record_duration)
        user_input_layout.addWidget(self.record_button)

        
        user_inputs.setLayout(user_input_layout)

        # labels of the Plot Channels (right)
        
        label_column      =   QVBoxLayout()       
        ch_labels         =   QWidget()
        ch_labels.setLayout(label_column)

        for ch in range(channel_number):
            temp_label = QLabel("Ch " + str(channel_number - ch))
            temp_label.setFont(QFont("Arial", 13))            
            label_column.addWidget(temp_label)

        ##### Final Layout of the Acquisition Window
            
        layout            =   QHBoxLayout()       # The GUI Main Layout
        widget.setLayout(layout)                # setting this layout to the widget

        layout.addWidget(user_inputs)           # text edit goes in middle-left
        layout.addWidget(win   )                # plot window goes on right side, spanning 3 rows
        layout.addWidget(ch_labels)
        
        self.setCentralWidget(widget)           # setting this widget as central widget of the main window


    def change_IP(self, text):              # update the IP address if it respects the IPv4 format
        address = text.split('.')           # otherwise input a default IPv4 address 
        if len(address) == 4:
            invalid_cnt = 0
            for number in address:
                if not x.isdigit():         # if not a number, increment invalid cnt
                    invalid_cnt += 1
                i = int(x)                  # if a number, check if is in the range 0-255
                if i < 0 or i > 255:        
                    invalid_cnt += 1

        if(invalid_cnt == 0):
            UDP_IP = text
        else:
            print("Invalid IP Address")

    def change_PORT(self, text):            # function called to update port if the given info is valid
        if text.isdigit():              
            number = int(text)
            if(number > 0 and number < 65536):
                UDP_PORT = int(text)
            else:
                UDP_PORT = 1234
                print("Port number Invalid!. Port set to:", UDP_PORT)
        else:
            UDP_PORT = 1234
            print("Not a Number! Port set to:", UDP_PORT)

            
    def connect_armband(self, checked):
        
        if checked == True:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)         
            sock.connect((UDP_IP, UDP_PORT))# client PC uses this
            self.connection_status = True
            self.connect_button.setText("Disconnect Armband")
            print('Armband Connection Successful')
            self.label_connected.setText("Armband Status: Connected")
        else:
            self.connection_status = False
            self.connect_button.setText("Connect Armband")
            self.label_connected.setText("Armband Status: Disconnected")
            print('Armband Disconnected')

    def start_stop_acquisition(self, checked):
        if checked == True:
            self.acquire_status = True
            self.acquire_button.setText("Acquiring...")
        else:
            self.acquire_status = False
            self.acquire_button.setText("Start Acquisition")


    def file_rename(self, text):
        #self.record_name = text
        self.record_name_sgn.emit(text)
        #print("Record File Changed to: ", self.record_name, ".npy", sep = '') 
        
            
    def set_record_duration(self, text):
        try:
            self.record_duration = int(text)
            print("Record length Changed to: ", self.record_duration, " seconds", sep = '')
        except:
            self.record_duration = 0
            print("Duration must be a number!")

             
    def record(self, checked):
        if checked == True:
            print("Signal Acquisition Started!") 
            self.record_button.setEnabled(False)            # disable button during acquisition
            self.record_button.setText("Recording Data...")
            # init the number of packets to be recorded, given the user input    
            self.record_status          = True
            self.record_counter_sgn.emit(int(sample_rate * self.record_duration / sliding_window))
            self.record_on_sgn.emit(self.record_status)

    
    def record_done(self, done_flag):
        self.record_button.setText("Done")
        time.sleep(0.5)
        self.record_button.setEnabled(True)
        self.record_button.setText("Start Recording Data")
        self.record_status      = False
        
        
    def update(self, new_data):
        # keep track of where the last segment was updated
        range_lim_1 = sliding_window *  self.plot_step
        range_lim_2 = sliding_window * (self.plot_step + 1)
        # this is the function that receives data from Acquisition Thread
        # if conditions to share data are fulfilled, then this function passes data towards other functions too

        if(self.record_status == True):         # 
            self.mutex.lock()
            self.record_payload_sgn.emit(new_data)
            self.mutex.unlock()
    
        #self.mutex.lock()                       # send data towards the neural network thread
        #self.ml_payload_sgn.emit(new_data)
        #self.mutex.unlock()
                
        new_samples = new_data
        # update the buffers
        self.channel_buffers[0, range_lim_1 : range_lim_2] = new_samples[0 * sliding_window : 1 * sliding_window]
        self.channel_buffers[1, range_lim_1 : range_lim_2] = new_samples[1 * sliding_window : 2 * sliding_window]
        self.channel_buffers[2, range_lim_1 : range_lim_2] = new_samples[2 * sliding_window : 3 * sliding_window]
        self.channel_buffers[3, range_lim_1 : range_lim_2] = new_samples[3 * sliding_window : 4 * sliding_window]
        self.channel_buffers[4, range_lim_1 : range_lim_2] = new_samples[4 * sliding_window : 5 * sliding_window]
        self.channel_buffers[5, range_lim_1 : range_lim_2] = new_samples[5 * sliding_window : 6 * sliding_window]
        self.channel_buffers[6, range_lim_1 : range_lim_2] = new_samples[6 * sliding_window : 7 * sliding_window]
        self.channel_buffers[7, range_lim_1 : range_lim_2] = new_samples[7 * sliding_window : 8 * sliding_window]
                
        self.plot_step += 1
        if(self.plot_step > window_size/sliding_window - 1):
            self.plot_step = 0           # if the pointer exceeds the range, then start from beginning
            print('Plot_Time: ',(time.time_ns() - self.start_moment)/1000000, ' msec')
            self.flag = 1
            self.start_moment = time.time_ns()        
        # display the plot
        self.traces[0].setData(self.channel_buffers[0])
        self.traces[1].setData(self.channel_buffers[1])
        self.traces[2].setData(self.channel_buffers[2])
        self.traces[3].setData(self.channel_buffers[3])
        self.traces[4].setData(self.channel_buffers[4])
        self.traces[5].setData(self.channel_buffers[5])
        self.traces[6].setData(self.channel_buffers[6])
        self.traces[7].setData(self.channel_buffers[7])    

    def gesture_result(self, gesture_number):
        # punem o imagine cu gestul detect
        print('Gest facut:', gesture_number)
        if  (gesture_number == 0):
            self.label_arrow.setPixmap(self.gesture_0_sign)
            #servo_con.write("0".encode())
        elif(gesture_number == 1):
            self.label_arrow.setPixmap(self.gesture_1_sign)
            #servo_con.write("1".encode())
        elif(gesture_number == 2):
            self.label_arrow.setPixmap(self.gesture_2_sign)
            #servo_con.write("2".encode())
        elif(gesture_number == 3):
            self.label_arrow.setPixmap(self.gesture_3_sign)
            #servo_con.write("3".encode())
        elif(gesture_number == 4):
            self.label_arrow.setPixmap(self.gesture_4_sign)
            #servo_con.write("4".encode())


#---------------------------------------------------
# Packet Retriever Thread

class portRead(QThread):
    
    sgn = pyqtSignal(list)        # this signal transmits an array of numbers

    def __init__(self):
        QThread.__init__(self)
        self.y = []                         #init the packet with the number of 
        self.moveToThread(self)
        self.mutex = QMutex()
        print("Init_Acquire_Thread")
        self.acquisitionTimer = QTimer()
        self.acquisitionTimer.moveToThread(self)
        self.acquisitionTimer.timeout.connect(self.retrievePacket)

    def run(self):
        self.acquisitionTimer.start(250) #(int(packet_size / 4))    # 4 packets per second means 250ms space between
        loop = QEventLoop()
        loop.exec_()

    def retrievePacket(self):
        print("Acquiring...")

        try:
            sock.send(start_cmd)           # write the sample command for the next subroutine execution
            self.mutex.lock()
        
            self.y , addr = sock.recvfrom(packet_size)
            
        except socket.timeout:
            print('2 x timeout')
            sock.send(retry_cmd)
            try: 
                self.y , addr = sock.recvfrom(packet_size)
            except:
                print('2 x err')
                pass
            
        temp = list(map(int,self.y))

        if(len(temp) != packet_size):
            self.y = []#128 * np.ones(packet_size)
            self.mutex.unlock()
            print('nema')
            return
            
        self.sgn.emit(temp)
        self.mutex.unlock()
        
#---------------------------------------------------
# Packet Recorder Thread

class dataRecord(QThread):              # a thread is created,

    done_sgn = pyqtSignal(bool)         # signal that stops recording

    recordState = False
    packet_count = 20

    storage_pointer = 0
    storage_buffers = 128 * np.ones(( channel_number , offset_val * packet_count), dtype=np.uint8)
                                        # define the buffer_length of the storage_buffer
    def __init__(self):                 # init is done in main window
        QThread.__init__(self)
        self.y = None
        self.moveToThread(self)
        self.file_count = 1
        self.mutex = QMutex()
        self.base_filename = 'record'
        self.filename = self.base_filename + str(self.file_count)

    def run(self):                      # it runs in non-blocking mode
        loop = QEventLoop()
        loop.exec_()

    def set_filename(self, filename):
        self.filename = filename + '.npy'
        base_filename = filename + '.npy'
        self.file_count = 1
        print("Record Filename Set to: ", self.filename)

    def toggle_on_off(self, state):
        self.recordState = state
        self.storage_pointer = 0
        print("Record Started")
        
    def set_packet_count(self, count):
        self.packet_count = count

        print("File expects a number of: ", self.packet_count, " packets")
        
        # there are 8 channels, and each packet delivers 128 samples per channel
        self.storage_buffers = 128 * np.ones(( channel_number , offset_val * self.packet_count), dtype=np.uint8)  # define the buffer_length of the storage_buffer
  
    def recorder(self, new_data):                 # when a button is pressed, start recording

        if(self.recordState == True):

            self.mutex.lock()
            new_samples = new_data
            self.mutex.unlock()
            
            range_lim_1 = 128 *  self.storage_pointer           # just like the sliding window, iterate through buffer until completion
            range_lim_2 = 128 * (self.storage_pointer + 1)
    
            self.storage_buffers[0, range_lim_1 : range_lim_2] = new_samples[0 * sliding_window : 1 * sliding_window]
            self.storage_buffers[1, range_lim_1 : range_lim_2] = new_samples[1 * sliding_window : 2 * sliding_window]
            self.storage_buffers[2, range_lim_1 : range_lim_2] = new_samples[2 * sliding_window : 3 * sliding_window]
            self.storage_buffers[3, range_lim_1 : range_lim_2] = new_samples[3 * sliding_window : 4 * sliding_window]
            self.storage_buffers[4, range_lim_1 : range_lim_2] = new_samples[4 * sliding_window : 5 * sliding_window]
            self.storage_buffers[5, range_lim_1 : range_lim_2] = new_samples[5 * sliding_window : 6 * sliding_window]
            self.storage_buffers[6, range_lim_1 : range_lim_2] = new_samples[6 * sliding_window : 7 * sliding_window]
            self.storage_buffers[7, range_lim_1 : range_lim_2] = new_samples[7 * sliding_window : 8 * sliding_window]
                    
            self.storage_pointer += 1         

            if(self.storage_pointer == self.packet_count):      # when the buffer is filled, write it in the file
                
                self.storage_pointer = 0
                with open(self.filename, 'wb') as f:
                #with open(self.filename) as f:
                    np.save(f, self.storage_buffers)

                print("Record Complete. File Name: ", self.filename)
                # for every successful record, automatically increment the file name
                
                self.filename = self.base_filename + str(self.file_count)
                self.file_count += 1
                
                self.recordState = False                    # when all packets were received and written to file, stop recording
                self.done_sgn.emit(True)        
                
#---------------------------------------------------
# Machine Learning Classifier Thread

class  ML_classifier(QThread):

    doneSignal = pyqtSignal(int)                            # this thread returns the name of the gesture performed
    window_count = 0
    scores = [0,0,0,0,0]            # there is a score vector. A number of signal windows are being classified
                                    # Each classification result increments the score for the respective gesture
                                    # The gesture with highest score is the one displayed

    window     =   offset_val * np.ones((channel_number, sliding_window), dtype= 'float')

    
    def __init__(self):                                     # init is done in main window

        QThread.__init__(self)
        self.moveToThread(self)
        tf.keras.utils.disable_interactive_logging()

        self.model_time = tf.keras.models.load_model('model_time.hdf5')  # path to the ML folder
        self.maximum = np.load('normalization.npy')
        
    def run(self):                                          # it runs in non-blocking mode
        loop = QEventLoop()
        loop.exec_()

    def analyze(self, new_data):

        
        
        self.window[0, : ] = np.asarray(new_data[0 * sliding_window : 1 * sliding_window], dtype = 'float') 
        self.window[1, : ] = np.asarray(new_data[1 * sliding_window : 2 * sliding_window], dtype = 'float') 
        self.window[2, : ] = np.asarray(new_data[2 * sliding_window : 3 * sliding_window], dtype = 'float') 
        self.window[3, : ] = np.asarray(new_data[3 * sliding_window : 4 * sliding_window], dtype = 'float') 
        self.window[4, : ] = np.asarray(new_data[4 * sliding_window : 5 * sliding_window], dtype = 'float') 
        self.window[5, : ] = np.asarray(new_data[5 * sliding_window : 6 * sliding_window], dtype = 'float') 
        self.window[6, : ] = np.asarray(new_data[6 * sliding_window : 7 * sliding_window], dtype = 'float') 
        self.window[7, : ] = np.asarray(new_data[7 * sliding_window : 8 * sliding_window], dtype = 'float') 

        self.window -= 128         # center the samples in the range [-128:+127]
    
        print("Window:", np.shape(self.window), type(self.window[0]), type(self.window[0,0]))
        
        inp = np.empty(0)
        for channel in range(8):        # for each of the channels
            features = self.features_calc(self.window[channel])

            for index in range(8):      # 8 features for one channel
                features[index] /= self.maximum[index]
        
            inp = np.append(inp, features)
        inp = np.reshape(inp, (1, 64))

        print("NN_INPUT:", np.shape(inp),"TIP", type(inp))
        
        analysis_result = self.model_time.predict(inp)

        #self.doneSignal.emit(int(np.argmax(analysis_result)))
        
        print("rezultat predictie: ", analysis_result)
        self.scores[int(np.argmax(analysis_result))] += 1
        self.window_count += 1

        if self.window_count == 3:
            print("rezultate gesturi: ",self.scores)
            self.doneSignal.emit(self.scores.index(max(self.scores))) # emit the gesture with highest score
            self.window_count = 0
            self.scores = [0,0,0,0,0]
        
    def features_calc(self, w):
        MAV = self.meanAbsoluteValue(w)
        SSC = self.slopeSignChanges(w)
        ZCR = self.zeroCrossRate(w)
        WL = self.waveformLength(w)
        Skewness = self.skewness(w)
        RMS = self.rootMeanSquare(w)
        Hjorth = self.hjorthActivity(w)
        iEMG = self.integratedEMG(w)
        return np.array([MAV, SSC, ZCR, WL, Skewness, RMS, Hjorth, iEMG])
    
    def meanAbsoluteValue(self, signal):
        return np.mean(np.abs(signal))

    def slopeSignChanges(self, signal):
        diff = signal[1:] - signal[:-1]
        return np.sum((diff[:-1] * diff[1:]) < 0)

    def zeroCrossRate(self, signal):
        return np.sum((signal[:-1] * signal[1:]) < 0)

    def waveformLength(self, signal): # lungimea formei semnalului
        return np.sum(np.abs(signal[1:] - signal[:-1]))

    def skewness(self, signal):
        std = np.std(signal)
        mean = np.mean(signal)
        cen = signal - mean
        return np.mean((cen / (std + 1e-3)) ** 3)

    def rootMeanSquare(self, signal):
        return np.sqrt(np.mean(signal ** 2))
    
    def hjorthActivity(self, signal): # varianta semnalului
        mean = np.mean(signal)
        cen = signal - mean
        return np.mean(cen ** 2)

    def integratedEMG(self, signal):
        return np.sum(np.abs(signal))   
    
    
        
#-----------------------------------------------------------------------------
#------------------------------- Main Loop -----------------------------------
app = QApplication(sys.argv)
 
# create the instance of our Window

window = Window()

if __name__ == '__main__':
# start the app
    sys.exit(app.exec())

