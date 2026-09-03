import pygame
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
from collections import deque

# ====================== 1. 全局超参数配置 ======================
WIDTH, HEIGHT = 900, 600
FPS = 60
LR = 1e-4
GAMMA = 0.95
MEMORY_CAPACITY = 10000
BATCH_SIZE = 64
EPSILON = 0.9  # 探索率

# 状态/动作维度（固定长度，见 get_state）
STATE_DIM = 46
ACTION_DIM = 6  # 0待机 1上 2下 3左 4右 5职业技能

# 火球最多同时存在的数量（用于状态向量固定槽位）
MAX_FIREBALLS = 8

# 立绘透明度（半透明，避免遮挡下方单位/弹道/范围圈）
SPRITE_ALPHA = 160

# 终局阶段：Boss 血量低于该比例时切换到 map2
FINAL_STAGE_HP_RATIO = 0.3

# 阶段切换暗红闪屏持续帧数
STAGE_FLASH_FRAMES = 18

# ====================== 2. 职业配置表 ======================
# 4 个职业的差异化属性：血量/移速/射程/每tick伤害/技能冷却/颜色/短名
CLASS_ORDER = ['warrior', 'mage', 'archer', 'healer']
CLASS_PARAMS = {
    'warrior': dict(name='WAR', color=(0, 180, 255), hp=150, speed=2.6,
                    attack_range=45,  attack_damage=0.5,  skill_cd=40),
    'mage':    dict(name='MAG', color=(160, 80, 255), hp=60,  speed=2.2,
                    attack_range=240, attack_damage=0.35, skill_cd=50),
    'archer':  dict(name='ARC', color=(30, 220, 100), hp=85,  speed=3.2,
                    attack_range=260, attack_damage=0.25, skill_cd=25),
    'healer':  dict(name='HEAL', color=(255, 220, 60), hp=70, speed=2.8,
                    attack_range=40,  attack_damage=0.05, skill_cd=35),
}

# 难度配置：boss_hp/boss_dmg/boss_speed/fireball_rate 作用于 Boss，unit_hp 作用于单位
DIFFICULTY_CONFIG = {
    '简单': dict(boss_hp=0.7, boss_dmg=0.7, boss_speed=0.9, fireball_rate=0.7, unit_hp=1.2),
    '普通': dict(boss_hp=1.0, boss_dmg=1.0, boss_speed=1.0, fireball_rate=1.0, unit_hp=1.0),
    '困难': dict(boss_hp=1.3, boss_dmg=1.3, boss_speed=1.2, fireball_rate=1.3, unit_hp=0.9),
}

# ====================== 3. DQN 神经网络定义 ======================
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.net(x)


# 强化学习智能体
class RLAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.eval_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.memory = deque(maxlen=MEMORY_CAPACITY)
        self.epsilon = EPSILON

    def choose_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        if np.random.uniform() < self.epsilon:
            action = np.random.randint(0, self.action_dim)
        else:
            actions = self.eval_net(state)
            action = torch.max(actions, 1)[1].data.numpy()[0]
        return action

    def store_transition(self, s, a, r, s_):
        self.memory.append((s, a, r, s_))

    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        s_batch = torch.FloatTensor([t[0] for t in batch])
        a_batch = torch.LongTensor([[t[1]] for t in batch])
        r_batch = torch.FloatTensor([[t[2]] for t in batch])
        s_next_batch = torch.FloatTensor([t[3] for t in batch])

        q_eval = self.eval_net(s_batch).gather(1, a_batch)
        q_next = self.target_net(s_next_batch).detach()
        q_target = r_batch + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)

        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# ====================== 4. 单位 / 火球 / Boss ======================
class Unit:
    def __init__(self, cls, x, y, hp_scale=1.0):
        p = CLASS_PARAMS[cls]
        self.cls = cls
        self.name = p['name']
        self.color = p['color']
        self.x = x
        self.y = y
        self.max_hp = p['hp'] * hp_scale
        self.hp = self.max_hp
        self.speed = p['speed']
        self.attack_range = p['attack_range']
        self.attack_damage = p['attack_damage']
        self.skill_cd = p['skill_cd']
        self.radius = 22
        self.skill_cd_left = 0   # 技能剩余冷却
        self.taunt_left = 0      # 战士嘲讽剩余 tick
        self.atk_fx_cd = 0       # 普攻特效节流计时

    def take_damage(self, dmg):
        self.hp = max(0, self.hp - dmg)

    def tick(self):
        if self.skill_cd_left > 0:
            self.skill_cd_left -= 1
        if self.taunt_left > 0:
            self.taunt_left -= 1
        if self.atk_fx_cd > 0:
            self.atk_fx_cd -= 1

    def move_by_action(self, act):
        if act == 1:
            self.y -= self.speed
        elif act == 2:
            self.y += self.speed
        elif act == 3:
            self.x -= self.speed
        elif act == 4:
            self.x += self.speed
        # 边界限制
        self.x = max(22, min(WIDTH - 22, self.x))
        self.y = max(22, min(HEIGHT - 22, self.y))

    def use_skill(self, boss, ally, effects):
        """释放职业技能，返回 (造成伤害, 治疗量)。冷却中则返回 (0, 0)。"""
        if self.skill_cd_left > 0:
            return 0, 0
        self.skill_cd_left = self.skill_cd
        dmg, heal = 0, 0
        if self.cls == 'warrior':
            # 嘲讽重击：强制 Boss 目标为自己，近身造成高额伤害
            self.taunt_left = 40
            effects.append(Effect('ring', self.x, self.y, (255, 220, 60), 16, radius=30))
            if dist(self, boss) < self.attack_range + 60:
                dmg = 3.0
                boss.take_damage(dmg)
                effects.append(Effect('ring', boss.x, boss.y, self.color, 12, radius=30))
                effects.append(Effect('flash', boss.x, boss.y, self.color, 10, radius=24))
        elif self.cls == 'mage':
            # 火球：远距离高额爆发
            if dist(self, boss) < 400:
                dmg = 2.5
                boss.take_damage(dmg)
                effects.append(Effect('line', self.x, self.y, self.color, 8, end=(boss.x, boss.y)))
                effects.append(Effect('flash', boss.x, boss.y, (255, 140, 40), 12, radius=28))
        elif self.cls == 'archer':
            # 强力射击：中额爆发、短冷却
            if dist(self, boss) < self.attack_range + 60:
                dmg = 1.8
                boss.take_damage(dmg)
                effects.append(Effect('line', self.x, self.y, self.color, 8, end=(boss.x, boss.y)))
                effects.append(Effect('ring', boss.x, boss.y, self.color, 10, radius=20))
        elif self.cls == 'healer':
            # 治疗：回复血量最低的队友（不在身边则回复自己）
            if dist(self, ally) < 300 and ally.hp < ally.max_hp:
                target = ally
            else:
                target = self
            heal = min(15, target.max_hp - target.hp)
            target.hp += heal
            if heal > 0:
                effects.append(Effect('heal', target.x, target.y, (0, 255, 120), 18))
                effects.append(Effect('ring', target.x, target.y, (0, 255, 120), 12, radius=24))
        return dmg, heal

    def draw(self, screen, font, sprite):
        x, y = int(self.x), int(self.y)
        w, h = sprite.get_size()
        # 职业立绘居中绘制（半透明）
        sprite.set_alpha(SPRITE_ALPHA)
        screen.blit(sprite, (x - w // 2, y - h // 2))
        # 血条（立绘上方）
        pygame.draw.rect(screen, (255, 0, 0), (x - 22, y - h // 2 - 12, 44, 7))
        pygame.draw.rect(screen, (0, 255, 0), (x - 22, y - h // 2 - 12, 44 * (max(0, self.hp) / self.max_hp), 7))
        # 职业标签
        label = font.render(self.name, True, (255, 255, 255))
        screen.blit(label, (x - label.get_width() // 2, y - h // 2 - 30))


class Fireball:
    def __init__(self, x, y, vx, vy):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.radius = 6
        self.damage = 12
        self.dead = False

    def update(self):
        self.x += self.vx
        self.y += self.vy

    def off_screen(self):
        return self.x < -20 or self.x > WIDTH + 20 or self.y < -20 or self.y > HEIGHT + 20

    def draw(self, screen):
        pygame.draw.circle(screen, (255, 120, 0), (int(self.x), int(self.y)), self.radius)


class Effect:
    """临时特效：扩散环 / 淡出闪光 / 弹道线 / 治疗十字"""
    def __init__(self, kind, x, y, color, lifetime, radius=20, end=None):
        self.kind = kind
        self.x = x
        self.y = y
        self.color = color
        self.lifetime = lifetime
        self.max_lifetime = lifetime
        self.radius = radius
        self.end = end

    def update(self):
        self.lifetime -= 1

    @property
    def alive(self):
        return self.lifetime > 0

    def draw(self, screen):
        t = self.lifetime / self.max_lifetime  # 1 -> 0
        if self.kind == 'ring':
            r = int(self.radius * (1 - t) + 3)
            pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), r, 2)
        elif self.kind == 'flash':
            r = self.radius
            surf = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
            pygame.draw.circle(surf, (*self.color, int(150 * t)), (r, r), r)
            screen.blit(surf, (int(self.x - r), int(self.y - r)))
        elif self.kind == 'line':
            if self.end is not None:
                ex, ey = self.end
                pygame.draw.line(screen, self.color, (int(self.x), int(self.y)), (int(ex), int(ey)), 2)
        elif self.kind == 'heal':
            dy = int(24 * (1 - t))
            cx, cy = int(self.x), int(self.y) - dy
            pygame.draw.rect(screen, self.color, (cx - 2, cy - 8, 4, 16))
            pygame.draw.rect(screen, self.color, (cx - 8, cy - 2, 16, 4))


class Boss:
    def __init__(self, x, y, diff=None):
        diff = diff or {}
        self.x = x
        self.y = y
        self.max_hp = 300 * diff.get('boss_hp', 1.0)
        self.hp = self.max_hp
        self.speed = 2.0 * diff.get('boss_speed', 1.0)
        self.radius = 28
        self.color = (255, 40, 40)
        self.melee_range = 50
        self.melee_damage = 0.3 * diff.get('boss_dmg', 1.0)
        self.fireball_rate = diff.get('fireball_rate', 1.0)
        self.enrage = False
        self.target = None
        self.retarget_timer = 0
        # AOE 预警圈
        self.aoe_cd = 120
        self.aoe_telegraph = None   # [x, y, 剩余 tick]
        self.aoe_radius = 90
        self.aoe_damage = 18 * diff.get('boss_dmg', 1.0)
        self.aoe_ticks = 45
        # 火球
        self.fireball_cd = 60

    @property
    def speed_now(self):
        return self.speed * (1.5 if self.enrage else 1.0)

    def take_damage(self, dmg):
        self.hp = max(0, self.hp - dmg)

    def update(self, units, fireballs, effects):
        """Boss 每帧逻辑，返回本帧对各单位造成的伤害 [(unit, dmg), ...]。"""
        hits = []

        # 阶段/狂暴
        self.enrage = self.hp <= self.max_hp * 0.5

        # 选目标（优先嘲讽单位，否则最近）
        self.retarget_timer += 1
        if self.retarget_timer > 25 or self.target is None:
            self.retarget_timer = 0
            taunting = [u for u in units if u.taunt_left > 0]
            if taunting:
                self.target = taunting[0]
            else:
                self.target = min(units, key=lambda u: dist(self, u))

        # 追击目标
        tx, ty = self.target.x, self.target.y
        if self.x < tx:
            self.x += self.speed_now
        elif self.x > tx:
            self.x -= self.speed_now
        if self.y < ty:
            self.y += self.speed_now
        elif self.y > ty:
            self.y -= self.speed_now
        self.x = max(self.radius, min(WIDTH - self.radius, self.x))
        self.y = max(self.radius, min(HEIGHT - self.radius, self.y))

        # 近战接触伤害
        for u in units:
            if dist(self, u) < self.melee_range:
                dmg = self.melee_damage * (1.5 if self.enrage else 1.0)
                u.take_damage(dmg)
                hits.append((u, dmg))

        # 范围 AOE（预警圈 → 爆炸）
        if self.aoe_telegraph is None:
            self.aoe_cd -= 1
            if self.aoe_cd <= 0:
                u = random.choice(units)
                self.aoe_telegraph = [u.x, u.y, self.aoe_ticks]
                effects.append(Effect('ring', u.x, u.y, (255, 120, 120), 12, radius=self.aoe_radius))
        else:
            self.aoe_telegraph[2] -= 1
            if self.aoe_telegraph[2] <= 0:
                tx, ty = self.aoe_telegraph[0], self.aoe_telegraph[1]
                effects.append(Effect('ring', tx, ty, (255, 60, 60), 14, radius=self.aoe_radius))
                effects.append(Effect('flash', tx, ty, (255, 90, 90), 12, radius=self.aoe_radius))
                for u in units:
                    if dist_pt(tx, ty, u) < self.aoe_radius:
                        dmg = self.aoe_damage * (1.5 if self.enrage else 1.0)
                        u.take_damage(dmg)
                        hits.append((u, dmg))
                self.aoe_telegraph = None
                self.aoe_cd = 200

        # 弹幕火球
        self.fireball_cd -= 1
        if self.fireball_cd <= 0:
            u = random.choice(units)
            dx, dy = u.x - self.x, u.y - self.y
            d = (dx * dx + dy * dy) ** 0.5 or 1.0
            spd = 5.0 * (1.3 if self.enrage else 1.0)
            fireballs.append(Fireball(self.x, self.y, dx / d * spd, dy / d * spd))
            self.fireball_cd = int((60 if self.enrage else 100) / self.fireball_rate)

        return hits

    def draw(self, screen, font, sprite):
        x, y = int(self.x), int(self.y)
        w, h = sprite.get_size()
        # Boss 立绘居中绘制（半透明）
        sprite.set_alpha(SPRITE_ALPHA)
        screen.blit(sprite, (x - w // 2, y - h // 2))
        # 狂暴时红色描边提示
        if self.enrage:
            pygame.draw.circle(screen, (255, 0, 0), (x, y), w // 2 + 6, 3)
        # Boss 血条（更宽，立绘上方）
        bw = 60
        pygame.draw.rect(screen, (255, 0, 0), (x - bw // 2, y - h // 2 - 14, bw, 8))
        pygame.draw.rect(screen, (0, 255, 0), (x - bw // 2, y - h // 2 - 14, bw * (max(0, self.hp) / self.max_hp), 8))
        label = font.render('BOSS', True, (255, 255, 255))
        screen.blit(label, (x - label.get_width() // 2, y - h // 2 - 32))
        # AOE 预警圈
        if self.aoe_telegraph is not None:
            tx, ty, _ = self.aoe_telegraph
            surf = pygame.Surface((self.aoe_radius * 2, self.aoe_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(surf, (255, 0, 0, 80), (self.aoe_radius, self.aoe_radius), self.aoe_radius)
            screen.blit(surf, (int(tx - self.aoe_radius), int(ty - self.aoe_radius)))
            pygame.draw.circle(screen, (255, 80, 80), (int(tx), int(ty)), self.aoe_radius, 2)


# ====================== 5. 状态 / 距离工具 ======================
def dist(u1, u2):
    return np.sqrt((u1.x - u2.x) ** 2 + (u1.y - u2.y) ** 2)


def dist_pt(x, y, u):
    return ((x - u.x) ** 2 + (y - u.y) ** 2) ** 0.5


def get_state(a1, a2, boss, fireballs):
    """返回固定长度 46 维状态向量。"""
    s = []
    # 两个单位：位置 + 血量 + 职业 one-hot
    for u in (a1, a2):
        s += [u.x / WIDTH, u.y / HEIGHT, u.hp / u.max_hp]
        s += [1.0 if u.cls == c else 0.0 for c in CLASS_ORDER]
    # Boss：位置 + 血量 + 狂暴 flag
    s += [boss.x / WIDTH, boss.y / HEIGHT, boss.hp / boss.max_hp, 1.0 if boss.enrage else 0.0]
    # AOE 预警圈：存在 + 位置 + 剩余时间
    if boss.aoe_telegraph is not None:
        s += [1.0, boss.aoe_telegraph[0] / WIDTH, boss.aoe_telegraph[1] / HEIGHT,
              boss.aoe_telegraph[2] / boss.aoe_ticks]
    else:
        s += [0.0, 0.0, 0.0, 0.0]
    # 火球固定槽位
    for i in range(MAX_FIREBALLS):
        if i < len(fireballs):
            fb = fireballs[i]
            s += [1.0, fb.x / WIDTH, fb.y / HEIGHT]
        else:
            s += [0.0, 0.0, 0.0]
    return s


# ====================== 5.5. 立绘加载 ======================
def load_sprites(unit_size=80, boss_size=120):
    """从 image 目录加载并缩放职业/Boss 立绘，返回 {cls/boss: Surface}。"""
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'image')
    sprites = {}
    for cls in CLASS_ORDER:
        img = pygame.image.load(os.path.join(base, cls + '.png')).convert_alpha()
        sprites[cls] = pygame.transform.smoothscale(img, (unit_size, unit_size))
    img = pygame.image.load(os.path.join(base, 'boss.png')).convert_alpha()
    sprites['boss'] = pygame.transform.smoothscale(img, (boss_size, boss_size))
    return sprites


def load_maps():
    """加载两张地图背景并缩放到窗口尺寸：normal=map1，final=map2。"""
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'image')
    maps = {}
    for key, name in (('normal', 'map1.png'), ('final', 'map2.png')):
        img = pygame.image.load(os.path.join(base, name)).convert()
        maps[key] = pygame.transform.smoothscale(img, (WIDTH, HEIGHT))
    return maps


# 普通攻击命中特效
def spawn_attack_fx(unit, boss, effects):
    """远程职业发射弹道线，近战职业只在命中点闪光。"""
    if unit.cls in ('mage', 'archer'):
        effects.append(Effect('line', unit.x, unit.y, unit.color, 6, end=(boss.x, boss.y)))
    effects.append(Effect('flash', boss.x, boss.y, unit.color, 8, radius=16))


# ====================== 6. 主训练循环 ======================
def train_episode(episode, screen, clock, font, sprites, maps, classes, diff, rl1, rl2):
    pygame.display.set_caption(f"协同DQN训练 对局:{episode}")

    # 使用首页选定的 2 个职业（固定队伍）
    clsA, clsB = classes[0], classes[1]
    hp_scale = diff.get('unit_hp', 1.0)
    agentA = Unit(clsA, 120, 200, hp_scale=hp_scale)
    agentB = Unit(clsB, 120, 400, hp_scale=hp_scale)
    boss = Boss(WIDTH - 150, HEIGHT // 2, diff=diff)

    fireballs = []
    effects = []
    is_final_stage = False     # 防止反复切图
    stage_flash = 0            # 阶段切换暗红闪屏剩余帧数
    flash_overlay = pygame.Surface((WIDTH, HEIGHT))
    flash_overlay.fill((120, 0, 0))  # 暗红遮罩
    done = False
    total_reward_A = 0.0
    total_reward_B = 0.0

    state = get_state(agentA, agentB, boss, fireballs)
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        # DQN 选择动作
        actA = rl1.choose_action(state)
        actB = rl2.choose_action(state)

        # 单位行动 + 冷却递减
        agentA.move_by_action(actA)
        agentB.move_by_action(actB)
        agentA.tick()
        agentB.tick()

        rewardA = 0.0
        rewardB = 0.0

        # 自动攻击（进入射程即造成伤害）
        if dist(agentA, boss) < agentA.attack_range:
            boss.take_damage(agentA.attack_damage)
            rewardA += agentA.attack_damage * 5
            if agentA.atk_fx_cd == 0:
                spawn_attack_fx(agentA, boss, effects)
                agentA.atk_fx_cd = 12
        if dist(agentB, boss) < agentB.attack_range:
            boss.take_damage(agentB.attack_damage)
            rewardB += agentB.attack_damage * 5
            if agentB.atk_fx_cd == 0:
                spawn_attack_fx(agentB, boss, effects)
                agentB.atk_fx_cd = 12

        # 职业技能（动作 5）
        if actA == 5:
            dmg, heal = agentA.use_skill(boss, agentB, effects)
            rewardA += dmg * 5 + heal * 2
        if actB == 5:
            dmg, heal = agentB.use_skill(boss, agentA, effects)
            rewardB += dmg * 5 + heal * 2

        # 终局阶段切换：Boss 血量 <= 30% 切到 map2（仅触发一次）
        if not is_final_stage and boss.hp <= boss.max_hp * FINAL_STAGE_HP_RATIO:
            is_final_stage = True
            stage_flash = STAGE_FLASH_FRAMES

        # 协作：残血队友掩护
        if agentA.hp < 30:
            if dist(agentA, agentB) < 200:
                rewardB += 25
            else:
                rewardB -= 20
        if agentB.hp < 30:
            if dist(agentA, agentB) < 200:
                rewardA += 25
            else:
                rewardA -= 20

        # Boss 机制（近战/AOE），返回受击列表用于惩罚
        hits = boss.update([agentA, agentB], fireballs, effects)
        for u, dmg in hits:
            effects.append(Effect('flash', u.x, u.y, (255, 60, 60), 8, radius=18))
            if u is agentA:
                rewardA -= dmg * 0.5
            else:
                rewardB -= dmg * 0.5

        # 火球更新 + 命中判定
        for fb in fireballs:
            fb.update()
            for u in (agentA, agentB):
                if dist(fb, u) < fb.radius + u.radius:
                    u.take_damage(fb.damage)
                    effects.append(Effect('flash', u.x, u.y, (255, 120, 0), 10, radius=20))
                    if u is agentA:
                        rewardA -= 15
                    else:
                        rewardB -= 15
                    fb.dead = True
                    break
        fireballs = [fb for fb in fireballs if not fb.dead and not fb.off_screen()]

        # 胜负判定
        if boss.hp <= 0:
            rewardA += 100
            rewardB += 100
            done = True
        elif agentA.hp <= 0 and agentB.hp <= 0:
            rewardA -= 100
            rewardB -= 100
            done = True

        state_next = get_state(agentA, agentB, boss, fireballs)
        rl1.store_transition(state, actA, rewardA, state_next)
        rl2.store_transition(state, actB, rewardB, state_next)
        rl1.learn()
        rl2.learn()
        state = state_next
        total_reward_A += rewardA
        total_reward_B += rewardB

        # 更新特效并清理过期项
        for e in effects:
            e.update()
        effects = [e for e in effects if e.alive]

        # 渲染（先铺地图背景：终局用 map2）
        screen.blit(maps['final'] if is_final_stage else maps['normal'], (0, 0))
        agentA.draw(screen, font, sprites[agentA.cls])
        agentB.draw(screen, font, sprites[agentB.cls])
        boss.draw(screen, font, sprites['boss'])
        for fb in fireballs:
            fb.draw(screen)
        for e in effects:
            e.draw(screen)
        # 阶段切换暗红全屏闪屏（淡出）
        if stage_flash > 0:
            flash_overlay.set_alpha(int(150 * stage_flash / STAGE_FLASH_FRAMES))
            screen.blit(flash_overlay, (0, 0))
            stage_flash -= 1
        pygame.display.flip()
        clock.tick(FPS)

    print(f"对局{episode} | {agentA.name}奖励:{round(total_reward_A, 1)} | "
          f"{agentB.name}奖励:{round(total_reward_B, 1)} | Boss剩余HP:{round(boss.hp, 1)}")


# ====================== 7. 训练入口 ======================
def run_game(classes, difficulty='普通'):
    """从首页启动训练：classes 为选定的 2 个职业，difficulty 为难度。"""
    diff = DIFFICULTY_CONFIG.get(difficulty, DIFFICULTY_CONFIG['普通'])
    rl1 = RLAgent(STATE_DIM, ACTION_DIM)
    rl2 = RLAgent(STATE_DIM, ACTION_DIM)

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 18)
    sprites = load_sprites(unit_size=80, boss_size=120)
    maps = load_maps()

    TRAIN_EPISODES = 80
    for ep in range(1, TRAIN_EPISODES + 1):
        train_episode(ep, screen, clock, font, sprites, maps, classes, diff, rl1, rl2)
        # 每 15 轮同步一次目标网络
        if ep % 15 == 0:
            rl1.target_net.load_state_dict(rl1.eval_net.state_dict())
            rl2.target_net.load_state_dict(rl2.eval_net.state_dict())
    pygame.quit()
    print("训练全部完成！")


if __name__ == "__main__":
    # 独立运行时：默认使用 战士+治疗 组合、普通难度
    run_game(['warrior', 'healer'], '普通')
