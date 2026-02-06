'use client';

import { useState, useRef, useEffect } from 'react';
import { Send, Search, MessageSquare, AlertTriangle, Lightbulb, User, Shield, Plus, Check } from 'lucide-react';
import { chat, ChatResponse } from '@/lib/api';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  mode?: string;
  conflicts?: string[];
  suggestions?: string[];
  // 可沉淀的内容
  settleable?: SettleableItem[];
}

// 可沉淀项
interface SettleableItem {
  id: string;
  content: string;
  settled: boolean;
}

interface NodeInfo {
  id: string;
  name: string;
}

interface ChatPanelProps {
  selectedNodeId?: string | null;
  selectedNodeName?: string;
  nodesList?: NodeInfo[];
  onNewNode?: (node: { name: string; content: string; rules: string[] }) => void;
  onSettleContent?: (content: string, type: 'rule' | 'description' | 'subnode', targetNodeId?: string) => void;
  initialMessages?: Message[];  // 从父组件传入的历史消息
  onMessagesChange?: (messages: Message[]) => void;  // 消息变化时的回调
  currentMode?: ModeType;  // 当前模式（由父组件控制）
  onModeChange?: (mode: ModeType) => void;  // 模式变化时的回调
}

// 模式配置
const modeConfig = {
  chat: { 
    icon: MessageSquare, 
    label: '对话', 
    color: '#6366f1',
    description: '自由讨论世界观'
  },
  audit: { 
    icon: Search, 
    label: '查漏', 
    color: '#10b981',
    description: '发现设定的缺漏和矛盾'
  },
  validate: { 
    icon: Shield, 
    label: '校验', 
    color: '#ef4444',
    description: '校验设定的科学/逻辑合理性'
  },
  character: { 
    icon: User, 
    label: '角色', 
    color: '#ec4899',
    description: '角色弧光与事件推动分析'
  },
};

export type ModeType = keyof typeof modeConfig;

// 解析 AI 回复中的可沉淀内容
function parseSettleableContent(content: string): { cleanContent: string; items: SettleableItem[] } {
  const items: SettleableItem[] = [];
  let cleanContent = content;
  
  // 匹配 📌 开头的行（可沉淀内容）
  const pinPattern = /📌\s*(.+?)(?=\n|$)/g;
  let match;
  let index = 0;
  
  while ((match = pinPattern.exec(content)) !== null) {
    items.push({
      id: `settle-${Date.now()}-${index++}`,
      content: match[1].trim(),
      settled: false,
    });
  }
  
  // 也匹配 【可沉淀】 块中的内容
  const blockPattern = /【可沉淀】\s*([\s\S]*?)(?=【|$)/g;
  while ((match = blockPattern.exec(content)) !== null) {
    const blockContent = match[1];
    const linePattern = /[•\-\*]\s*(.+?)(?=\n|$)/g;
    let lineMatch;
    while ((lineMatch = linePattern.exec(blockContent)) !== null) {
      const itemContent = lineMatch[1].trim();
      // 避免重复
      if (!items.some(i => i.content === itemContent)) {
        items.push({
          id: `settle-${Date.now()}-${index++}`,
          content: itemContent,
          settled: false,
        });
      }
    }
  }
  
  return { cleanContent, items };
}

export default function ChatPanel({ 
  selectedNodeId, 
  selectedNodeName, 
  nodesList = [],
  onNewNode, 
  onSettleContent,
  initialMessages = [],
  onMessagesChange,
  currentMode = 'chat',
  onModeChange
}: ChatPanelProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<'world' | 'character'>('world');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  
  // 使用父组件传入的 mode
  const mode = currentMode;
  const setMode = (newMode: ModeType) => {
    if (onModeChange) {
      onModeChange(newMode);
    }
  };
  
  // 沉淀弹窗状态
  const [showSettleModal, setShowSettleModal] = useState(false);
  const [settleContent, setSettleContent] = useState('');
  const [settleType, setSettleType] = useState<'rule' | 'description' | 'subnode'>('rule');
  const [settleTargetId, setSettleTargetId] = useState<string>('');
  
  // 用于控制是否需要滚动到底部
  const shouldScrollToBottomRef = useRef(false);
  const lastMessageCountRef = useRef(0);

  // 初始化消息（从父组件传入的历史）
  // 由于使用了 key prop，组件会在切换时重新挂载，所以直接设置即可
  useEffect(() => {
    setMessages(initialMessages || []);
    // 切换对话框时，滚动到底部
    shouldScrollToBottomRef.current = true;
  }, [initialMessages]);

  // 消息变化时通知父组件（使用 ref 来避免循环依赖）
  const onMessagesChangeRef = useRef(onMessagesChange);
  useEffect(() => {
    onMessagesChangeRef.current = onMessagesChange;
  }, [onMessagesChange]);
  
  useEffect(() => {
    if (onMessagesChangeRef.current && messages.length > 0) {
      onMessagesChangeRef.current(messages);
    }
  }, [messages]);

  // 滚动到底部的逻辑
  useEffect(() => {
    // 只在以下情况滚动到底部：
    // 1. shouldScrollToBottomRef 为 true（组件首次加载或切换模式）
    // 2. 消息数量增加（发送了新消息）
    const messageCountIncreased = messages.length > lastMessageCountRef.current;
    
    if (shouldScrollToBottomRef.current || messageCountIncreased) {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
      shouldScrollToBottomRef.current = false;
    }
    
    lastMessageCountRef.current = messages.length;
  }, [messages]);

  // 当切换 tab 时，同步切换模式
  const handleTabChange = (tab: 'world' | 'character') => {
    setActiveTab(tab);
    if (tab === 'character') {
      setMode('character');
    } else if (mode === 'character') {
      // 从角色 tab 切回世界观 tab，默认切到对话模式
      setMode('chat');
    }
    // 切换 tab 时滚动到底部
    shouldScrollToBottomRef.current = true;
  };

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      mode,
    };

    // 构建历史消息（只取最近 10 条，不包含当前要发送的）
    const historyForApi = messages
      .slice(-10)
      .map(m => ({
        role: m.role,
        content: m.content,
      }));

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      // 发送时带上历史消息
      const response: ChatResponse = await chat(
        input, 
        mode, 
        selectedNodeId || undefined,
        historyForApi
      );
      
      // 解析可沉淀内容
      const { cleanContent, items } = parseSettleableContent(response.response);
      
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: cleanContent,
        conflicts: response.conflicts,
        suggestions: response.suggestions,
        settleable: items.length > 0 ? items : undefined,
      };

      setMessages(prev => [...prev, assistantMessage]);

      if (response.new_nodes && response.new_nodes.length > 0 && onNewNode) {
        response.new_nodes.forEach(node => {
          onNewNode({
            name: node.name,
            content: node.content,
            rules: node.rules,
          });
        });
      }
    } catch (error) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: `❌ 出错了: ${error instanceof Error ? error.message : '未知错误'}`,
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // 打开沉淀弹窗
  const openSettleModal = (content: string) => {
    setSettleContent(content);
    setSettleType('rule');
    setSettleTargetId(selectedNodeId || '');
    setShowSettleModal(true);
  };

  // 确认沉淀
  const confirmSettle = (messageId: string, itemId: string) => {
    if (!onSettleContent) return;
    
    // 执行沉淀
    onSettleContent(settleContent, settleType, settleTargetId || undefined);
    
    // 标记已沉淀
    setMessages(prev => prev.map(msg => {
      if (msg.id === messageId && msg.settleable) {
        return {
          ...msg,
          settleable: msg.settleable.map(item => 
            item.id === itemId ? { ...item, settled: true } : item
          ),
        };
      }
      return msg;
    }));
    
    setShowSettleModal(false);
  };

  const getAvailableModes = (): ModeType[] => {
    if (activeTab === 'character') {
      return ['character'];
    }
    return ['chat', 'audit', 'validate'];
  };

  const availableModes = getAvailableModes();

  return (
    <div className="chat-panel h-full flex flex-col">
      {/* 沉淀确认弹窗 */}
      {showSettleModal && (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50">
          <div className="bg-[var(--secondary)] p-6 rounded-xl border border-[var(--border)] max-w-md w-full mx-4">
            <h3 className="text-lg font-semibold mb-4">添加到节点</h3>
            
            <div className="bg-black/30 p-3 rounded-lg mb-4 text-sm">
              {settleContent}
            </div>
            
            {/* 目标节点选择 */}
            <div className="mb-4">
              <label className="text-sm text-gray-400 mb-2 block">添加到：</label>
              <select 
                value={settleTargetId}
                onChange={(e) => setSettleTargetId(e.target.value)}
                className="input-field text-sm"
              >
                <option value="">新建节点</option>
                {nodesList.map(node => (
                  <option key={node.id} value={node.id}>
                    {node.name} {node.id === selectedNodeId ? '(当前选中)' : ''}
                  </option>
                ))}
              </select>
            </div>
            
            {/* 添加类型 */}
            <div className="mb-6">
              <label className="text-sm text-gray-400 mb-2 block">添加为：</label>
              <div className="flex gap-2">
                <button
                  onClick={() => setSettleType('rule')}
                  className={`flex-1 py-2 px-3 rounded-lg text-sm transition-all ${
                    settleType === 'rule' 
                      ? 'bg-[var(--primary)] text-white' 
                      : 'bg-black/30 text-gray-400 hover:text-white'
                  }`}
                >
                  规则
                </button>
                <button
                  onClick={() => setSettleType('description')}
                  className={`flex-1 py-2 px-3 rounded-lg text-sm transition-all ${
                    settleType === 'description' 
                      ? 'bg-[var(--primary)] text-white' 
                      : 'bg-black/30 text-gray-400 hover:text-white'
                  }`}
                >
                  描述
                </button>
                <button
                  onClick={() => setSettleType('subnode')}
                  className={`flex-1 py-2 px-3 rounded-lg text-sm transition-all ${
                    settleType === 'subnode' 
                      ? 'bg-[var(--primary)] text-white' 
                      : 'bg-black/30 text-gray-400 hover:text-white'
                  }`}
                >
                  子节点
                </button>
              </div>
            </div>
            
            <div className="flex gap-3">
              <button
                onClick={() => setShowSettleModal(false)}
                className="btn-secondary flex-1"
              >
                取消
              </button>
              <button
                onClick={() => {
                  // 找到对应的 message 和 item
                  const msgWithItem = messages.find(m => 
                    m.settleable?.some(i => i.content === settleContent && !i.settled)
                  );
                  const item = msgWithItem?.settleable?.find(i => i.content === settleContent && !i.settled);
                  if (msgWithItem && item) {
                    confirmSettle(msgWithItem.id, item.id);
                  } else {
                    // 直接沉淀
                    if (onSettleContent) {
                      onSettleContent(settleContent, settleType, settleTargetId || undefined);
                    }
                    setShowSettleModal(false);
                  }
                }}
                className="btn-primary flex-1"
              >
                确认添加
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Tab 切换 */}
      <div className="flex border-b border-[var(--border)]">
        <button
          onClick={() => handleTabChange('world')}
          className={`flex-1 py-3 text-sm font-medium transition-all ${
            activeTab === 'world'
              ? 'text-[var(--primary)] border-b-2 border-[var(--primary)]'
              : 'text-gray-400 hover:text-white'
          }`}
        >
          🌍 世界观
        </button>
        <button
          onClick={() => handleTabChange('character')}
          className={`flex-1 py-3 text-sm font-medium transition-all ${
            activeTab === 'character'
              ? 'text-[var(--primary)] border-b-2 border-[var(--primary)]'
              : 'text-gray-400 hover:text-white'
          }`}
        >
          👤 角色
        </button>
      </div>

      {/* 头部 */}
      <div className="p-4 border-b border-[var(--border)]">
        <h2 className="text-lg font-semibold mb-1">
          {activeTab === 'world' ? '世界观顾问' : '角色顾问'}
        </h2>
        <p className="text-xs text-gray-500 mb-2">
          {activeTab === 'world' 
            ? '帮你发现设定的缺漏，点击 📌 可添加到节点' 
            : '帮你完善角色设定，检查角色对事件的推动作用'}
        </p>
        
        {selectedNodeName && (
          <div className="text-sm text-gray-400 mb-3">
            当前节点: <span className="text-[var(--primary)]">{selectedNodeName}</span>
          </div>
        )}
        
        {/* 模式选择 */}
        <div className="flex flex-wrap gap-2">
          {availableModes.map((key) => {
            const config = modeConfig[key];
            const Icon = config.icon;
            const isActive = mode === key;
            return (
              <button
                key={key}
                onClick={() => setMode(key)}
                className={`flex items-center gap-1 px-3 py-1.5 rounded-full text-xs transition-all ${
                  isActive 
                    ? 'text-white' 
                    : 'text-gray-400 hover:text-white'
                }`}
                style={{ 
                  background: isActive ? config.color : 'transparent',
                  border: `1px solid ${isActive ? config.color : 'var(--border)'}` 
                }}
                title={config.description}
              >
                <Icon className="w-3 h-3" />
                {config.label}
              </button>
            );
          })}
        </div>
        
        <p className="text-xs text-gray-500 mt-2">
          {modeConfig[mode].description}
        </p>
      </div>

      {/* 消息列表 */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && (
          <div className="text-center text-gray-500 mt-10">
            {activeTab === 'world' ? (
              <>
                <Search className="w-12 h-12 mx-auto mb-4 opacity-30" />
                <p>输入你的世界观设定</p>
                <p className="text-sm mt-2">AI 会帮你发现缺漏，你可以选择添加到节点</p>
              </>
            ) : (
              <>
                <User className="w-12 h-12 mx-auto mb-4 opacity-30" />
                <p>输入角色的设定信息</p>
                <p className="text-sm mt-2">我会帮你分析角色的完整度和事件推动作用</p>
              </>
            )}
          </div>
        )}
        
        {messages.map((msg) => (
          <div key={msg.id} className={`chat-message ${msg.role}`}>
            {msg.mode && msg.role === 'user' && (
              <div className="text-xs opacity-70 mb-1">
                [{modeConfig[msg.mode as ModeType]?.label || msg.mode}]
              </div>
            )}
            <div className="whitespace-pre-wrap">{msg.content}</div>
            
            {/* 可沉淀内容 */}
            {msg.settleable && msg.settleable.length > 0 && (
              <div className="mt-3 space-y-2">
                <div className="text-xs text-gray-400 flex items-center gap-1">
                  <Plus className="w-3 h-3" />
                  点击可添加到节点：
                </div>
                {msg.settleable.map((item) => (
                  <button
                    key={item.id}
                    onClick={() => !item.settled && openSettleModal(item.content)}
                    disabled={item.settled}
                    className={`w-full text-left p-3 rounded-lg border transition-all text-sm ${
                      item.settled
                        ? 'bg-green-500/10 border-green-500/30 text-green-400'
                        : 'bg-[var(--primary)]/10 border-[var(--primary)]/30 hover:border-[var(--primary)] hover:bg-[var(--primary)]/20'
                    }`}
                  >
                    <div className="flex items-start gap-2">
                      {item.settled ? (
                        <Check className="w-4 h-4 mt-0.5 flex-shrink-0" />
                      ) : (
                        <span className="text-base">📌</span>
                      )}
                      <span className={item.settled ? 'line-through opacity-70' : ''}>
                        {item.content}
                      </span>
                    </div>
                  </button>
                ))}
              </div>
            )}
            
            {/* 冲突警告 */}
            {msg.conflicts && msg.conflicts.length > 0 && (
              <div className="mt-3 p-3 rounded-lg bg-red-500/10 border border-red-500/30">
                <div className="flex items-center gap-2 text-red-400 text-sm font-medium mb-2">
                  <AlertTriangle className="w-4 h-4" />
                  检测到冲突
                </div>
                <ul className="text-sm text-red-300 space-y-1">
                  {msg.conflicts.map((conflict, i) => (
                    <li key={i}>• {conflict}</li>
                  ))}
                </ul>
              </div>
            )}
            
            {/* 建议 */}
            {msg.suggestions && msg.suggestions.length > 0 && (
              <div className="mt-3 p-3 rounded-lg bg-yellow-500/10 border border-yellow-500/30">
                <div className="flex items-center gap-2 text-yellow-400 text-sm font-medium mb-2">
                  <Lightbulb className="w-4 h-4" />
                  建议
                </div>
                <ul className="text-sm text-yellow-300 space-y-1">
                  {msg.suggestions.slice(0, 5).map((suggestion, i) => (
                    <li key={i}>• {suggestion}</li>
                  ))}
                  {msg.suggestions.length > 5 && (
                    <li className="text-gray-500">...还有 {msg.suggestions.length - 5} 条</li>
                  )}
                </ul>
              </div>
            )}
          </div>
        ))}
        
        {isLoading && (
          <div className="chat-message assistant">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-[var(--primary)] rounded-full animate-pulse" />
              <div className="w-2 h-2 bg-[var(--primary)] rounded-full animate-pulse" style={{ animationDelay: '0.1s' }} />
              <div className="w-2 h-2 bg-[var(--primary)] rounded-full animate-pulse" style={{ animationDelay: '0.2s' }} />
              <span className="text-gray-400 ml-2">分析中...</span>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* 输入区域 */}
      <div className="p-4 border-t border-[var(--border)]">
        <div className="flex gap-2">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={
              activeTab === 'character' 
                ? '输入角色的设定信息（姓名、背景、性格、在故事中的作用等）...'
                : mode === 'audit' ? '描述你的世界观设定，我来帮你查漏补缺...' :
                  mode === 'validate' ? '输入要校验的设定...' :
                  '和我聊聊你的世界观...'
            }
            className="input-field flex-1 resize-none"
            rows={3}
            disabled={isLoading}
          />
          <button
            onClick={handleSend}
            disabled={isLoading || !input.trim()}
            className="btn-primary self-end disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Send className="w-5 h-5" />
          </button>
        </div>
      </div>
    </div>
  );
}
