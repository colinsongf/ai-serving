# -*- coding: utf-8 -*-
try:
    import configparser
except:
    import ConfigParser as configparser

import threading
import time

import schedule
from wai.logger import LOGGER
from wai.model_manager import deploy_model

DEPLOY_CONFIG_PATH = ""
MODELS_CONFIG_PATH = ""
SCHEDULE_CONFIG_PATH = ""


class ScheduleException(Exception): pass


def deploy_and_refresh(model_code, deploy_config_path, model_config_path):
    LOGGER.info("Thread: %s. Begin to do schedule deploy_and_refresh job, model code->%s, "
                "deploy_config_path -> %s, model_config_path -> %s", threading.current_thread(), model_code,
                deploy_config_path, model_config_path)
    try:
        deploy_model(model_code, deploy_config_path)
        LOGGER.info("model %s deploy and rfresh success!" % model_code)
    except Exception as ep:
        err_msg = "Thread: %s .Model %s schedule task failed! %s" % (threading.current_thread(), model_code, str(ep))
        LOGGER.error(err_msg)
        # raise ScheduleException(err_msg)


def get_schedule_job(count=1, period=None, time_at='03:00'):
    job = schedule.every(count)
    if count == 1:
        if period == 'day':
            job = job.day.at(time_at)
        elif period == 'hour':
            job = job.hour
        elif period == 'minute':
            job = job.minute
        elif period == 'second':
            job = job.second
        else:
            job = job.day.at(time_at)
    else:
        if period == 'day':
            job = job.days.at(time_at)
        elif period == 'hour':
            job = job.hours
        elif period == 'minute':
            job = job.minutes
        elif period == 'second':
            job = job.seconds
        else:
            raise ScheduleException("Invalid period " + period)
    return job


def get_daily_job(time_at, job_func):
    return schedule.every().days.at(time_at).do(job_func)


def run_threaded(model_code, deploy_config_path, model_config_path):
    job_thread = threading.Thread(target=deploy_and_refresh, args=(model_code, deploy_config_path, model_config_path,))
    job_thread.start()


def get_jobs(schedule_conf_path):
    config = ConfigParser()
    config.read(schedule_conf_path)
    model_codes = config.sections()
    jobs = {}
    for code in model_codes:
        period = config.get(code, 'period')
        count = config.getint(code, 'count')
        task_day_time = ""
        if period == 'day':
            task_day_time = config.get(code, 'time')
            jobs[code] = get_schedule_job(count=count, period=period, time_at=task_day_time)
            jobs[code].tag(code)
        else:
            jobs[code] = get_schedule_job(count=count, period=period)
            jobs[code].tag(code)
    return jobs


def init_config_jobs(schedule_conf_path):
    all_jobs = get_jobs(schedule_conf_path)
    for model_code, sjob in all_jobs.items():
        sjob.do(run_threaded, model_code, DEPLOY_CONFIG_PATH, MODELS_CONFIG_PATH)


def do_deploy_refresh_jobs(schedule_conf_path):
    init_config_jobs(schedule_conf_path)
    i = 0
    while 1:
        schedule.run_pending()
        time.sleep(1)


def clear_all_jobs():
    schedule.clear()


def clear_job(model_code):
    schedule.clear(model_code)


def test_schedule():
    print("test here ..")
    do_deploy_refresh_jobs(SCHEDULE_CONFIG_PATH)
