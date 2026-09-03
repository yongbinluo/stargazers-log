import os
import glob
import pygame

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'image')


def _find_bgm():
    """在 image 目录里查找第一个 mp3 作为背景音乐（避免硬编码中文文件名）。"""
    files = sorted(glob.glob(os.path.join(BASE, '*.mp3')))
    return files[0] if files else None


class BGMManager:
    """背景音乐管理：加载 / 播放(循环) / 停止 / 开关。"""
    def __init__(self, path=None, volume=0.5):
        self.path = path or _find_bgm()
        self.volume = volume
        self.playing = False

    def init(self):
        if not pygame.mixer.get_init():
            try:
                pygame.mixer.init()
            except pygame.error:
                # 无音频设备时静默降级，避免崩溃
                pass

    def play(self):
        if not self.path:
            return
        self.init()
        try:
            pygame.mixer.music.load(self.path)
            pygame.mixer.music.set_volume(self.volume)
            pygame.mixer.music.play(-1)   # -1 表示循环播放
            self.playing = True
        except pygame.error:
            self.playing = False

    def stop(self):
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()
        self.playing = False

    def toggle(self):
        if self.playing:
            self.stop()
        else:
            self.play()
        return self.playing


class BGMButton:
    """BGM 开关按钮：负责绘制与点击检测（文字需要中文字体，由调用方传入）。"""
    def __init__(self, rect):
        self.rect = pygame.Rect(rect)

    def handle_click(self, pos, manager):
        if self.rect.collidepoint(pos):
            manager.toggle()
            return True
        return False

    def draw(self, screen, font, manager):
        playing = manager.playing
        text = '音乐: 开' if playing else '音乐: 关'
        color = (0, 200, 120) if playing else (130, 130, 130)
        pygame.draw.rect(screen, (50, 50, 60), self.rect, border_radius=6)
        pygame.draw.rect(screen, color, self.rect, 2, border_radius=6)
        label = font.render(text, True, color)
        screen.blit(label, (self.rect.centerx - label.get_width() // 2,
                            self.rect.centery - label.get_height() // 2))
