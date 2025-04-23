import random
import math
from copy import deepcopy
import time

class AIPlayer:
    """
    AI 玩家 - 使用蒙特卡洛树搜索算法
    """

    def __init__(self, color):
        """
        玩家初始化
        :param color: 下棋方，'X' - 黑棋，'O' - 白棋
        """
        self.color = color
        self.opponent_color = 'O' if color == 'X' else 'X'
        # 减少时间限制，提高响应速度
        self.time_limit = 5  
        # 调整UCT参数，加快收敛速度
        self.C = 1.0  # 原来是1.4，降低探索倾向性
        # 添加早停阈值
        self.min_visits = 100  # 当根节点访问次数达到此值时可以提前返回结果
        # 添加缓存字典保存棋盘状态评估结果
        self.board_cache = {}

    def get_move(self, board):
        """
        根据当前棋盘状态获取最佳落子位置
        :param board: 棋盘
        :return: action 最佳落子位置, e.g. 'A1'
        """
        # ...现有代码...
        
        # 获取当前合法落子位置
        legal_actions = list(board.get_legal_actions(self.color))
        if not legal_actions:
            return None
        if len(legal_actions) == 1:
            return legal_actions[0]
            
        # 添加简单启发式评估，对角落位置优先选择
        corner_positions = ['A1', 'A8', 'H1', 'H8']
        for corner in corner_positions:
            if corner in legal_actions:
                return corner
                
        # 创建根节点
        root = Node(board=deepcopy(board), parent=None, action=None, color=self.color)
        
        # 运行蒙特卡洛树搜索
        action = self.monte_carlo_tree_search(root)
        return action

    def monte_carlo_tree_search(self, root):
        """
        蒙特卡洛树搜索算法主体，添加早停机制
        :param root: 根节点
        :return: 最佳落子位置
        """
        start_time = time.time()
        iterations = 0
        
        # 在时间限制内尽可能多地进行搜索迭代
        while time.time() - start_time < self.time_limit:
            # 添加早停条件
            if root.visits > self.min_visits and iterations > 100:
                break
                
            # 选择
            leaf = self.select(root)
            
            # 扩展
            if not leaf.is_terminal and not leaf.is_fully_expanded:
                child = self.expand(leaf)
                # 模拟
                reward = self.simulate(child)
                # 反向传播
                self.backpropagate(child, reward)
            else:
                # 如果叶子节点是终端节点或已完全扩展，直接进行模拟
                reward = self.simulate(leaf)
                # 反向传播
                self.backpropagate(leaf, reward)
                
            iterations += 1
            
        # 选择访问次数最多的子节点
        best_child = self.get_best_child(root, 0)  # 设置探索参数为0，纯粹基于访问次数
        return best_child.action


    def select(self, node):
        """
        选择阶段 - 使用UCB公式选择最有潜力的节点
        :param node: 当前节点
        :return: 选中的叶子节点
        """
        while not node.is_terminal:
            if not node.is_fully_expanded:
                return node
            node = self.get_best_child(node, self.C)
        return node

    def expand(self, node):
        """
        扩展阶段 - 创建新的子节点
        :param node: 当前节点
        :return: 新创建的子节点
        """
        # 获取所有可能的动作
        tried_actions = [child.action for child in node.children]
        legal_actions = list(node.board.get_legal_actions(node.color))
        
        # 过滤掉已经尝试过的动作
        untried_actions = [action for action in legal_actions if action not in tried_actions]
        
        if not untried_actions:
            node.is_fully_expanded = True
            return self.get_best_child(node, self.C)
        
        # 随机选择一个未尝试的动作
        action = random.choice(untried_actions)
        
        # 创建新的棋盘状态
        new_board = deepcopy(node.board)
        new_board._move(action, node.color)
        
        # 下一个玩家的颜色
        next_color = 'O' if node.color == 'X' else 'X'
        
        # 检查下一个玩家是否有合法动作
        if not list(new_board.get_legal_actions(next_color)):
            # 如果下一个玩家没有合法动作，可能需要跳过该玩家的回合
            # 检查当前玩家是否还有其他合法动作
            if not list(new_board.get_legal_actions(node.color)):
                # 游戏结束
                is_terminal = True
            else:
                # 当前玩家继续
                next_color = node.color
                is_terminal = False
        else:
            is_terminal = False
        
        # 创建并返回新节点
        child = Node(
            board=new_board,
            parent=node,
            action=action,
            color=next_color,
            is_terminal=is_terminal
        )
        node.children.append(child)
        
        # 如果所有可能的动作都已尝试，将节点标记为完全扩展
        if len(node.children) == len(legal_actions):
            node.is_fully_expanded = True
            
        return child

    def simulate(self, node):
        """
        模拟阶段 - 使用轻量级快速模拟策略
        :param node: 要模拟的节点
        :return: 模拟结果的奖励值
        """
        # 使用棋盘状态的哈希值作为缓存键
        board_state = str(node.board._board)
        if board_state in self.board_cache:
            return self.board_cache[board_state]
            
        # 创建一个临时棋盘副本
        board_copy = deepcopy(node.board)
        current_color = node.color
        
        # 减少最大模拟深度，加速模拟过程
        max_depth = 20  # 原来是50
        
        # 快速模拟算法 - 使用简单的随机策略
        depth = 0
        while depth < max_depth:
            # 检查是否有合法动作
            legal_actions = list(board_copy.get_legal_actions(current_color))
            if not legal_actions:
                # 切换玩家
                current_color = 'O' if current_color == 'X' else 'X'
                legal_actions = list(board_copy.get_legal_actions(current_color))
                if not legal_actions:
                    # 游戏结束
                    break
            
            # 优先选择角落位置，提高模拟质量
            corner_positions = ['A1', 'A8', 'H1', 'H8']
            action = None
            for corner in corner_positions:
                if corner in legal_actions:
                    action = corner
                    break
                    
            # 如果没有角落位置，随机选择
            if not action:
                action = random.choice(legal_actions)
                
            board_copy._move(action, current_color)
            
            # 切换玩家
            current_color = 'O' if current_color == 'X' else 'X'
            depth += 1
        
        # 计算模拟结果
        winner, diff = board_copy.get_winner()
        
        # 确定奖励值
        if winner == 2:  # 平局
            reward = 0.5
        elif (winner == 0 and self.color == 'X') or (winner == 1 and self.color == 'O'):
            # AI赢了
            reward = 1.0
        else:
            # AI输了
            reward = 0.0
            
        # 缓存结果
        self.board_cache[board_state] = reward
        return reward

    def backpropagate(self, node, reward):
        """
        反向传播阶段 - 更新节点统计信息
        :param node: 当前节点
        :param reward: 奖励值
        """
        while node:
            node.visits += 1
            # 如果当前节点的父节点的颜色与AI相同，则使用奖励值；否则使用(1-奖励值)
            if node.parent and node.parent.color == self.color:
                node.value += reward
            else:
                node.value += 1 - reward
            node = node.parent

    def get_best_child(self, node, exploration_weight):
        """
        根据UCB公式选择最佳子节点
        :param node: 当前节点
        :param exploration_weight: 探索权重
        :return: 最佳子节点
        """
        # 定义一个比较函数用于选择最佳子节点
        def ucb_score(child):
            # 避免除以零
            if child.visits == 0:
                return float('inf')
            
            # UCB公式
            exploitation = child.value / child.visits
            exploration = exploration_weight * math.sqrt(2 * math.log(node.visits) / child.visits)
            return exploitation + exploration
        
        # 如果当前节点颜色与AI相同，选择最大UCB值；否则选择最小UCB值
        if node.color == self.color:
            return max(node.children, key=ucb_score)
        else:
            return min(node.children, key=lambda child: -ucb_score(child))


class Node:
    """
    蒙特卡洛树搜索中的节点
    """
    def __init__(self, board, parent, action, color, is_terminal=False):
        """
        节点初始化
        :param board: 棋盘状态
        :param parent: 父节点
        :param action: 到达此节点的动作
        :param color: 当前节点玩家的颜色
        :param is_terminal: 是否为终端节点
        """
        self.board = board
        self.parent = parent
        self.action = action
        self.color = color  # 表示轮到哪种颜色下棋
        self.children = []
        self.visits = 0  # 访问次数
        self.value = 0.0  # 节点价值
        self.is_fully_expanded = False
        self.is_terminal = is_terminal