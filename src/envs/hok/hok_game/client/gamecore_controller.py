#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import path, sys
from pathlib import Path
folder = path.Path(__file__).abspath()
current_dir = Path(__file__).parent ## 获得当前目录路径
sys.path.append(str(current_dir))# 如果使用相对路径，并且添加当前目录的上两级目录

import time
import requests
from conf.config import GC_CONFIG


class GameCoreController:
    """
    GameCoreController 通过 http 请求给 ugc_game_core_server 启动/停止 gamecore
    """
    def __init__(self, game_id, logger, battlesrv_port):
        self.game_id = game_id
        self.ai_server_port = battlesrv_port
        self.ugc_project_id = GC_CONFIG["ugc_project_id"]
        self.ugc_level_name = GC_CONFIG["level_name"]
        self.game_core_server_endpoint = GC_CONFIG["endpoint"]
        self.retry_times = GC_CONFIG["retry_times"]
        self.retry_times_sleep_seconds = GC_CONFIG["retry_sleep_seconds"]
        self.logger = logger

    def start_game(self):
        """
            Description: 发送 http 请求给 ugc_game_core_server 启动 gamecore
            ----------

            Return: 成功返回 True, 重试超时失败返回 False
            ----------
        """
        new_game_req = {
            "game_id": self.game_id,
            "ai_server_port": self.ai_server_port,
            "ugc_project_id": self.ugc_project_id,
            "ugc_level_name": self.ugc_level_name,
        }

        endpoint_url = f"http://{self.game_core_server_endpoint}/ugc/newGame"

        try:
            # gamecore 确保返回 ok 时是一定启动成功的, 增加重试次数
            resp = requests.post(url=endpoint_url, json=new_game_req)
            retry_times = 0
            while retry_times < self.retry_times and resp.status_code != 200:
                resp = requests.post(url=endpoint_url, json=new_game_req)
                time.sleep(self.retry_times_sleep_seconds)
                retry_times += 1

            if retry_times >= self.retry_times:
                self.logger.error(f"start_game response code: {resp.status_code}, response content: {resp.content}")
                return False

            self.logger.info(f'Succeed to start Game using SGame.exe, new_game_req: {new_game_req}')
            return True

        except Exception as ex:
            self.logger.error(
                f'failed to start Game using SGame.exe, new_game_req: {new_game_req}, error is {str(ex)}')
            return False

    def stop_game(self):
        """
            Description: 发送 http 请求给 ugc_game_core_server 停止gamecore
            ----------

            Return: 成功返回 True, 重试超时失败返回 False
            ----------
        """
        stop_game_req = {
            "game_id": self.game_id,
            "ai_server_port": self.ai_server_port,
        }
        endpoint_url = f"http://{self.game_core_server_endpoint}/ugc/stopGame"

        try:
            resp = requests.post(url=endpoint_url, json=stop_game_req)
            retry_times = 0
            while retry_times < self.retry_times and resp.status_code != 200:
                resp = requests.post(url=endpoint_url, json=stop_game_req)
                time.sleep(self.retry_times_sleep_seconds)
                retry_times += 1

            if retry_times >= self.retry_times:
                self.logger.error(f"stop_game response code: {resp.status_code}, response content: {resp.content}")
                return False

            self.logger.info(f'Succeed to stop Game using SGame.exe, stop_game_req: {stop_game_req}')
            return True

        except Exception as ex:
            self.logger.error(
                f'failed to stop Game using SGame.exe, stop_game_req: {stop_game_req}, error is {str(ex)}')
            return False
