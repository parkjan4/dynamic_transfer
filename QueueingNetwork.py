import numpy as np

rdm_service = np.random.RandomState()
rdm_arrival = np.random.RandomState()
rdm_departure = np.random.RandomState()

class Simulation:

    def __init__(self):
        self.HospitalNames = []
        self.Frequency = 'daily'
        self.Queues = []
        self.Resources = []
        self.Entities = []
        self.EventList = []
        self.waitingTimes = []
        self.Clock = 0
        self.StartDate = None
        self.Policy = None
        self.EndTime = np.inf
        self.endSim = False
        self.entityId = 0
        self.List = []
        self.Queue = []
        self.timeList = []
        self.Departures = []
        self.ArrivalsFromToday = []
        self.opt_params = None
        self.params_for_today = None
        self.WardArrivalsToday = 0
        self.WardDeparturesToday = 0
        self.lastSolvedTransfers = []
        
    
    def generate_service_time(self, patient):
    # Generates patient service time (LOS)
        base_process_service_time = rdm_service.exponential(base_rate)
        base_process_end_time =  base_process_service_time + self.opt_params.getnonInverted(patient.hospitalIndex, start_time)
        invertedTimeExp = self.opt_params.getInverted(patient.hospitalIndex, base_process_end_time)
        
    
    def timeDependentRegularAcuteServiceInversion(self, patient):
    # used for robustness experiments; checking lognormal service times
    
        base_rate = 1
        base_sd = np.mean([nonCovid_ward_CV_Fridays,nonCovid_ward_CV_Mondays,nonCovid_ward_CV_Weekdays,nonCovid_ward_CV_Weekends])
        logmean = np.log(np.square(base_rate) / np.sqrt(np.square(base_sd) + np.square(base_rate)))
        logstdev = np.sqrt(np.log(1 + np.square(base_sd)))
        start_time = self.Clock
        base_process_service_time = rdm_service.lognormal(logmean,logstdev)
        base_process_end_time =  base_process_service_time + self.opt_params.getnonInverted(patient.hospitalIndex, start_time)
        invertedTimeLog = self.opt_params.getInverted(patient.hospitalIndex, base_process_end_time)

        # base_process_service_time = rdm_service.exponential(base_rate)
        # base_process_end_time =  base_process_service_time + self.opt_params.getnonInverted(patient.hospitalIndex, start_time)
        # invertedTimeExp = self.opt_params.getInverted(patient.hospitalIndex, base_process_end_time)

        # print('nonCOVID ward',invertedTimeExp - start_time,invertedTimeLog - start_time)
        return (invertedTimeLog - start_time)