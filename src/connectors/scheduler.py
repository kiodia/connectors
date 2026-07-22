'''
Created on 2 mai 2020

@author: vankomme
'''

# pragma: no cover
from datetime import datetime, timedelta
from calendar import monthrange
import json
import logging as log


class Scheduler(object):
    '''
    classdocs
    '''
    planning = {}  # date List of tasks
    tasks = {}   #"google" "offset"  "number"  "source"
    seelink_schedule =""

    def __init__(self): # pragma: no cover
        '''
        Constructor
        '''
       
       
    def remaining_day(self,day) : # pragma: no cover
        date = datetime.strptime(day,'%Y%m%d')
        y= date.year
        m= date.month
        d= date.day
        start, end = monthrange(y, m)
        return end-d
        
          
       
    def get_next_month (self, day): # pragma: no cover
        '''
        Get the first day of the next month with the format %Y%m%d
        
        Parameter
        ---------
        
        day : string
            the date with the format %Y%m%d
            
        Returns
        -------
        
        String : the first day of the next month with the format %Y%m%d
        
        '''
        date = datetime.strptime(day,'%Y%m%d')
        date = (date.replace(day=1) + timedelta(days=32)).replace(day=1)  
        return date.strftime("%Y%m%d") 
       
     
    def get_next_day (self, day): # pragma: no cover
        '''
        Get the next day with the format %Y%m%d
        
        Parameter
        ---------
        
        day : string
            the date with the format %Y%m%d
            
        Returns
        -------
        
        String : dates of the day after with the format %Y%m%d
        
        '''
        date = datetime.strptime(day,'%Y%m%d')
        date += timedelta(days=1)        
        return date.strftime("%Y%m%d")
     
     
    def create_task_per_day(self,date_of_tasks,tasks): # pragma: no cover
        self.planning[date_of_tasks]= tasks
        return
            
    def create_task_per_month(self, month, tasks):   # pragma: no cover
        return  
    
    
    def write_schedule(self,schedule_path): # pragma: no cover
        with open(schedule_path, 'w',encoding='utf-8') as f:
            json.dump(self.planning, f) #, ensure_ascii=False, indent=4)
        
        f.close()

      
    def read_scheduled_tasks(self,schedule_path): # pragma: no cover
        with open(schedule_path, 'r') as f:
            self.planning = json.load(f)
            
        f.close()    
        return  
        
    def get_task_of_today(self,schedule_path): # pragma: no cover
        '''
        Read the scheduled task from file and return the one of today
        
        '''
        self.read_scheduled_tasks(schedule_path)
        now = datetime.now()
        today = now.strftime("%Y%m%d")
        try:
            tasks = self.planning[today]
        except KeyError:
            return None
        
        return tasks