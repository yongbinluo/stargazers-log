import os
import sys
import glob
import importlib.util
import pygame
import bgm

# ====================== 1. 常量与布局 ======================
WIDTH, HEIGHT = 900, 600
FPS = 60

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'image')

CLASS_ORDER = ['warrior', 'mage', 'archer', 'healer']
CLASS_LABELS = {'warrior': '战士', 'mage': '法师', 'archer': '弓箭手', 'healer': '治疗'}

DIFFICULTIES = ['简单', '普通', '困难']
DEFAULT_DIFFICULTY = '普通'

# 颜色
WHITE = (255, 255, 255)
BLACK = (20, 20, 20)
GRAY = (130, 130, 130)
DARK_GRAY = (70, 70, 70)
GREEN = (0, 200, 120)
GREEN_DISABLED = (80, 100, 90)
HIGHLIGHT = (255, 220, 0)
BLUE = (0, 180, 255)

# 布局（可按需调整，匹配 ui.png 的空白区域）
CARD_W, CARD_H = 110, 130      # 职业卡片尺寸
CARD_Y = 85                    # 卡片顶部 y
CARD_GAP = 40                  # 卡片间距
DIFF_Y = 320                   # 难度选择顶部 y
DIFF_W, DIFF_H = 150, 50       # 难度按钮尺寸
DIFF_GAP = 30                  # 难度按钮间距
START_RECT = pygame.Rect(300, 450, 300, 60)   # 开始游戏按钮
BACK_RECT = pygame.Rect(20, 20, 100, 40)      # 返回按钮
BGM_RECT = pygame.Rect(WIDTH - 130, 20, 110, 40)   # BGM 开关按钮


# 常见中文字体文件路径（直接按文件加载，绕开 pygame sysfont 的注册表解析 bug）
FONT_CANDIDATES = [
    r'C:\Windows\Fonts\msyh.ttc',    # 微软雅黑
    r'C:\Windows\Fonts\msyh.ttf',
    r'C:\Windows\Fonts\simhei.ttf',  # 黑体
    r'C:\Windows\Fonts\simsun.ttc',  # 宋体
    r'C:\Windows\Fonts\Deng.ttf',    # 等线
]


def load_font(size):
    """优先按文件路径加载中文字体，找不到则退回默认字体（可能不显示中文）。"""
    for path in FONT_CANDIDATES:
        if os.path.isfile(path):
            try:
                return pygame.font.Font(path, size)
            except Exception:
                continue
    return pygame.font.Font(None, size)


# ====================== 2. 首页场景 ======================
class HomeScene:
    def __init__(self, screen):
        self.screen = screen
        self.clock = pygame.time.Clock()
        self.font_big = load_font(28)
        self.font_mid = load_font(22)
        self.font_small = load_font(18)

        self.background = self._load_background()
        self.card_sprites = self._load_card_sprites()

        self.selected = set()            # 已选职业
        self.difficulty = DEFAULT_DIFFICULTY
        self.result = None               # run() 的返回结果
        self.running = True

        self.bgm = bgm.BGMManager()      # 背景音乐
        self.bgm.play()
        self.bgm_button = bgm.BGMButton(BGM_RECT)

        self.card_rects = self._build_card_rects()
        self.diff_rects = self._build_diff_rects()

    def _load_background(self):
        """读取外部背景图 ui.png 并缩放到窗口尺寸。"""
        img = pygame.image.load(os.path.join(BASE, 'ui.png')).convert()
        return pygame.transform.smoothscale(img, (WIDTH, HEIGHT))

    def _load_card_sprites(self):
        sprites = {}
        for cls in CLASS_ORDER:
            img = pygame.image.load(os.path.join(BASE, cls + '.png')).convert_alpha()
            sprites[cls] = pygame.transform.smoothscale(img, (80, 80))
        return sprites

    def _build_card_rects(self):
        total = 4 * CARD_W + 3 * CARD_GAP
        x0 = (WIDTH - total) // 2
        rects = {}
        for i, cls in enumerate(CLASS_ORDER):
            x = x0 + i * (CARD_W + CARD_GAP)
            rects[cls] = pygame.Rect(x, CARD_Y, CARD_W, CARD_H)
        return rects

    def _build_diff_rects(self):
        total = len(DIFFICULTIES) * DIFF_W + (len(DIFFICULTIES) - 1) * DIFF_GAP
        x0 = (WIDTH - total) // 2
        rects = []
        for i, d in enumerate(DIFFICULTIES):
            x = x0 + i * (DIFF_W + DIFF_GAP)
            rects.append((d, pygame.Rect(x, DIFF_Y, DIFF_W, DIFF_H)))
        return rects

    def handle_click(self, pos):
        # 职业卡片：点击切换选中
        for cls in CLASS_ORDER:
            if self.card_rects[cls].collidepoint(pos):
                if cls in self.selected:
                    self.selected.discard(cls)
                elif len(self.selected) < 2:
                    self.selected.add(cls)
                return
        # 难度：点击切换（单选）
        for d, rect in self.diff_rects:
            if rect.collidepoint(pos):
                self.difficulty = d
                return
        # BGM 开关
        if self.bgm_button.handle_click(pos, self.bgm):
            return
        # 开始游戏：未选满 2 个职业时不可点击（置灰）
        if START_RECT.collidepoint(pos) and len(self.selected) == 2:
            self.result = {'classes': [c for c in CLASS_ORDER if c in self.selected],
                           'difficulty': self.difficulty}
            self.running = False
            return
        # 返回
        if BACK_RECT.collidepoint(pos):
            self.result = None
            self.running = False

    def draw(self):
        # 背景
        self.screen.blit(self.background, (0, 0))

        # 职业卡片
        for cls in CLASS_ORDER:
            rect = self.card_rects[cls]
            selected = cls in self.selected
            pygame.draw.rect(self.screen, (40, 40, 50), rect, border_radius=8)
            # 选中高亮边框
            pygame.draw.rect(self.screen,
                             HIGHLIGHT if selected else GRAY,
                             rect, 4 if selected else 2, border_radius=8)
            sprite = self.card_sprites[cls]
            self.screen.blit(sprite, (rect.centerx - 40, rect.y + 10))
            label = self.font_small.render(CLASS_LABELS[cls], True, WHITE)
            self.screen.blit(label, (rect.centerx - label.get_width() // 2, rect.bottom - 28))

        # 难度选择
        for d, rect in self.diff_rects:
            selected = d == self.difficulty
            pygame.draw.rect(self.screen, BLUE if selected else (50, 50, 60), rect, border_radius=8)
            pygame.draw.rect(self.screen,
                             HIGHLIGHT if selected else GRAY, rect, 2, border_radius=8)
            label = self.font_mid.render(d, True, WHITE if selected else GRAY)
            self.screen.blit(label, (rect.centerx - label.get_width() // 2,
                                     rect.centery - label.get_height() // 2))

        # 开始游戏按钮（未选满 2 个职业时置灰）
        enabled = len(self.selected) == 2
        pygame.draw.rect(self.screen, GREEN if enabled else GREEN_DISABLED, START_RECT, border_radius=10)
        pygame.draw.rect(self.screen, WHITE if enabled else DARK_GRAY, START_RECT, 2, border_radius=10)
        label = self.font_big.render('开始游戏', True, WHITE if enabled else (180, 180, 180))
        self.screen.blit(label, (START_RECT.centerx - label.get_width() // 2,
                                 START_RECT.centery - label.get_height() // 2))

        # 选择数量提示
        hint = f"已选 {len(self.selected)}/2 个职业"
        hint_label = self.font_small.render(hint, True, HIGHLIGHT if enabled else GRAY)
        self.screen.blit(hint_label, (START_RECT.centerx - hint_label.get_width() // 2,
                                      START_RECT.bottom + 10))

        # 返回按钮
        pygame.draw.rect(self.screen, (60, 60, 70), BACK_RECT, border_radius=6)
        pygame.draw.rect(self.screen, GRAY, BACK_RECT, 2, border_radius=6)
        label = self.font_small.render('返回', True, WHITE)
        self.screen.blit(label, (BACK_RECT.centerx - label.get_width() // 2,
                                 BACK_RECT.centery - label.get_height() // 2))

        # BGM 开关按钮
        self.bgm_button.draw(self.screen, self.font_small, self.bgm)

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return None
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.handle_click(event.pos)
            self.draw()
            pygame.display.flip()
            self.clock.tick(FPS)
        return self.result


# ====================== 3. 启动自检 ======================
def check_environment():
    """启动前环境自检：依赖库 / 资源文件 / 中文字体 / 背景音乐。"""
    errors, warnings = [], []

    # 1) 依赖库
    for mod, hint in (('pygame', 'pip install pygame'),
                      ('numpy', 'pip install numpy'),
                      ('torch', 'pip install torch')):
        if importlib.util.find_spec(mod) is None:
            errors.append(f"缺少依赖库 {mod}（请执行 {hint}）")

    # 2) 资源文件
    required = ['ui.png', 'map1.png', 'map2.png', 'boss.png'] + [c + '.png' for c in CLASS_ORDER]
    for name in required:
        if not os.path.isfile(os.path.join(BASE, name)):
            errors.append(f"缺少资源文件 image/{name}")

    # 3) 背景音乐（非关键）
    if not glob.glob(os.path.join(BASE, '*.mp3')):
        warnings.append("未找到 mp3 背景音乐，将静音运行")

    # 4) 中文字体（非关键）
    if load_font(16).metrics('开')[0] is None:
        warnings.append("未找到中文字体，界面中文可能显示为方框")

    return errors, warnings


def _show_errors(errors, warnings):
    """打印问题清单，并尽力弹窗提示。"""
    for w in warnings:
        print(f"[警告] {w}")
    if not errors:
        return
    print("=" * 52)
    print("启动失败，检测到以下问题：")
    for e in errors:
        print("  ✗ " + e)
    print("=" * 52)
    try:
        import tkinter.messagebox as mb
        mb.showerror("启动失败", "\n".join(errors))
    except Exception:
        pass


# ====================== 4. 程序入口 ======================
def main():
    pygame.init()
    errors, warnings = check_environment()
    if errors:
        _show_errors(errors, warnings)
        pygame.quit()
        sys.exit(1)
    for w in warnings:
        print(f"[警告] {w}")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("游戏首页")
    scene = HomeScene(screen)
    result = scene.run()
    if result is None:
        pygame.quit()
        print("已退出首页")
        return None
    # 开始游戏：把所选职业与难度交给 coop_dqn_bot 训练
    import coop_dqn_bot
    coop_dqn_bot.run_game(result['classes'], result['difficulty'])
    return result


if __name__ == "__main__":
    main()
