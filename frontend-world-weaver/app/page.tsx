'use client';

import { useCallback, useState, useEffect, useRef } from 'react';
import ReactFlow, {
  Node,
  Edge,
  Controls,
  MiniMap,
  Background,
  useNodesState,
  useEdgesState,
  addEdge,
  Connection,
  BackgroundVariant,
  NodeTypes,
  NodeChange,
  EdgeChange,
} from 'reactflow';
import 'reactflow/dist/style.css';
import { Plus, Save, Upload, Globe, Trash2, Cloud, CloudOff } from 'lucide-react';
import WorldNodeComponent from '@/components/WorldNode';
import ChatPanel, { ModeType } from '@/components/ChatPanel';
import { 
  createWorld, getWorld, getNodes, createNode, deleteNode, updateNode,
  exportWorld, importWorld, WorldNode 
} from '@/lib/api';

// 注册自定义节点类型
const nodeTypes: NodeTypes = {
  worldNode: WorldNodeComponent,
};

// localStorage key
const STORAGE_KEY = 'world-weaver-data';

// 对话消息类型（与 ChatPanel 中的 Message 兼容）
interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  mode?: string;
  timestamp?: number;
  conflicts?: string[];
  suggestions?: string[];
  settleable?: { id: string; content: string; settled: boolean }[];
}

// 对话历史类型（按节点存储）
interface Conversations {
  global: ChatMessage[];
  [nodeId: string]: ChatMessage[];
}

// 保存数据结构
interface SavedData {
  worldName: string;
  worldDescription: string;
  nodes: Node[];
  edges: Edge[];
  conversations: Conversations;  // 新增：对话历史
  savedAt: string;
}

// 沉淀内容类型
export interface SettleContent {
  content: string;
  type: 'rule' | 'description' | 'subnode';
}

// 初始节点
const initialNodes: Node[] = [];
const initialEdges: Edge[] = [];

// 最大保留的历史消息数
const MAX_HISTORY_MESSAGES = 10; // 保留最近 10 条（5 轮对话）

export default function WorldWeaverPage() {
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);
  const [worldName, setWorldName] = useState<string>('');
  const [worldDescription, setWorldDescription] = useState<string>('');
  const [isWorldLoaded, setIsWorldLoaded] = useState(false);
  const [showCreateWorld, setShowCreateWorld] = useState(false);
  const [newWorldName, setNewWorldName] = useState('');
  const [newWorldDesc, setNewWorldDesc] = useState('');
  
  // 对话历史
  const [conversations, setConversations] = useState<Conversations>({ global: [] });
  
  // 当前对话模式
  const [currentMode, setCurrentMode] = useState<ModeType>('chat');
  
  // 保存状态
  const [saveStatus, setSaveStatus] = useState<'saved' | 'saving' | 'unsaved'>('saved');
  const [lastSaved, setLastSaved] = useState<string>('');
  const saveTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const isInitialLoad = useRef(true);

  // 编辑节点弹窗状态
  const [showEditNode, setShowEditNode] = useState(false);
  const [editingNode, setEditingNode] = useState<Node | null>(null);
  const [editNodeName, setEditNodeName] = useState('');
  const [editNodeContent, setEditNodeContent] = useState('');
  const [editNodeRules, setEditNodeRules] = useState('');

  // ========== localStorage 操作 ==========
  
  const saveToLocal = useCallback(() => {
    if (!worldName) return;
    
    const data: SavedData = {
      worldName,
      worldDescription,
      nodes,
      edges,
      conversations,
      savedAt: new Date().toISOString(),
    };
    
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
      setLastSaved(new Date().toLocaleTimeString());
      setSaveStatus('saved');
      console.log('✅ [Storage] Saved to localStorage (with conversations)');
    } catch (error) {
      console.error('❌ [Storage] Save failed:', error);
      setSaveStatus('unsaved');
    }
  }, [worldName, worldDescription, nodes, edges, conversations]);

  const loadFromLocal = useCallback((): SavedData | null => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        const data = JSON.parse(saved) as SavedData;
        console.log('✅ [Storage] Loaded from localStorage:', data.worldName);
        return data;
      }
    } catch (error) {
      console.error('❌ [Storage] Load failed:', error);
    }
    return null;
  }, []);

  const clearLocal = useCallback(() => {
    localStorage.removeItem(STORAGE_KEY);
    console.log('🗑️ [Storage] Cleared localStorage');
  }, []);

  // ========== 自动保存 ==========
  
  const debouncedSave = useCallback(() => {
    if (isInitialLoad.current) return;
    
    setSaveStatus('saving');
    
    if (saveTimeoutRef.current) {
      clearTimeout(saveTimeoutRef.current);
    }
    
    saveTimeoutRef.current = setTimeout(() => {
      saveToLocal();
    }, 1000);
  }, [saveToLocal]);

  useEffect(() => {
    if (!isInitialLoad.current && worldName) {
      debouncedSave();
    }
  }, [nodes, edges, worldName, worldDescription, conversations, debouncedSave]);

  // ========== 初始加载 ==========
  
  useEffect(() => {
    const localData = loadFromLocal();
    
    if (localData && localData.worldName) {
      setWorldName(localData.worldName);
      setWorldDescription(localData.worldDescription || '');
      setNodes(localData.nodes || []);
      setEdges(localData.edges || []);
      setConversations(localData.conversations || { global: [] });
      setIsWorldLoaded(true);
      setLastSaved(new Date(localData.savedAt).toLocaleTimeString());
      
      createWorld(localData.worldName, localData.worldDescription || '').catch(() => {});
      
      console.log('✅ [Init] Restored from localStorage (with conversations)');
    } else {
      loadFromServer();
    }
    
    setTimeout(() => {
      isInitialLoad.current = false;
    }, 500);
  }, []);

  const loadFromServer = async () => {
    try {
      const world = await getWorld();
      setWorldName(world.name);
      setIsWorldLoaded(true);
      
      const { nodes: worldNodes } = await getNodes();
      const flowNodes: Node[] = worldNodes.map((node: WorldNode) => ({
        id: node.id,
        type: 'worldNode',
        position: { x: node.position_x, y: node.position_y },
        data: {
          name: node.name,
          node_type: node.node_type,
          content: node.content,
          rules: node.rules,
          tags: node.tags,
        },
      }));
      setNodes(flowNodes);
      
      const flowEdges: Edge[] = worldNodes
        .filter((node: WorldNode) => node.parent_id)
        .map((node: WorldNode) => ({
          id: `e-${node.parent_id}-${node.id}`,
          source: node.parent_id!,
          target: node.id,
          animated: true,
          style: { stroke: '#6366f1' },
        }));
      setEdges(flowEdges);
    } catch {
      setIsWorldLoaded(false);
      setShowCreateWorld(true);
    }
  };

  // ========== 节点操作 ==========

  const onConnect = useCallback(
    (params: Connection) => setEdges((eds) => addEdge({ 
      ...params, 
      animated: true,
      style: { stroke: '#6366f1' }
    }, eds)),
    [setEdges]
  );

  const onNodeClick = useCallback((_: React.MouseEvent, node: Node) => {
    setSelectedNode(node);
  }, []);

  // 点击背景取消选中节点
  const onPaneClick = useCallback(() => {
    setSelectedNode(null);
  }, []);

  // 双击节点进入编辑模式
  const onNodeDoubleClick = useCallback((_: React.MouseEvent, node: Node) => {
    setEditingNode(node);
    setEditNodeName(node.data.name || '');
    setEditNodeContent(node.data.content || '');
    setEditNodeRules((node.data.rules || []).join('\n'));
    setShowEditNode(true);
  }, []);

  // 保存编辑的节点
  const handleSaveEditNode = useCallback(() => {
    if (!editingNode) return;
    
    const newRules = editNodeRules
      .split('\n')
      .map(r => r.trim())
      .filter(r => r);
    
    // 更新节点
    setNodes(nds => nds.map(n => {
      if (n.id === editingNode.id) {
        return {
          ...n,
          data: {
            ...n.data,
            name: editNodeName || n.data.name,
            content: editNodeContent,
            rules: newRules,
          }
        };
      }
      return n;
    }));
    
    // 同步更新 selectedNode
    if (selectedNode?.id === editingNode.id) {
      setSelectedNode({
        ...editingNode,
        data: {
          ...editingNode.data,
          name: editNodeName || editingNode.data.name,
          content: editNodeContent,
          rules: newRules,
        }
      });
    }
    
    setShowEditNode(false);
    setEditingNode(null);
  }, [editingNode, editNodeName, editNodeContent, editNodeRules, setNodes, selectedNode]);

  const handleNodesChange = useCallback((changes: NodeChange[]) => {
    onNodesChange(changes);
  }, [onNodesChange]);

  const handleEdgesChange = useCallback((changes: EdgeChange[]) => {
    onEdgesChange(changes);
  }, [onEdgesChange]);

  const handleCreateWorld = async () => {
    if (!newWorldName.trim()) return;
    try {
      await createWorld(newWorldName, newWorldDesc);
      setWorldName(newWorldName);
      setWorldDescription(newWorldDesc);
      setIsWorldLoaded(true);
      setShowCreateWorld(false);
      setNewWorldName('');
      setNewWorldDesc('');
      
      setTimeout(() => {
        isInitialLoad.current = false;
        saveToLocal();
      }, 100);
    } catch (error) {
      console.error('创建世界失败:', error);
    }
  };

  const handleAddNode = async () => {
    const name = prompt('节点名称:');
    if (!name) return;
    
    const nodeType = prompt('节点类型 (world/region/race/character/magic/tech/religion/history/org/rule/item/creature/custom):', 'custom');
    
    const position = {
      x: Math.random() * 400,
      y: Math.random() * 400,
    };

    try {
      const result = await createNode({
        name,
        node_type: nodeType || 'custom',
        content: '',
        position_x: position.x,
        position_y: position.y,
        parent_id: selectedNode?.id || null,
      });

      const newNode: Node = {
        id: result.node.id,
        type: 'worldNode',
        position,
        data: {
          name: result.node.name,
          node_type: result.node.node_type,
          content: result.node.content,
          rules: result.node.rules,
          tags: result.node.tags,
        },
      };

      setNodes((nds) => [...nds, newNode]);

      if (selectedNode) {
        setEdges((eds) => [
          ...eds,
          {
            id: `e-${selectedNode.id}-${result.node.id}`,
            source: selectedNode.id,
            target: result.node.id,
            animated: true,
            style: { stroke: '#6366f1' },
          },
        ]);
      }
    } catch (error) {
      console.error('创建节点失败:', error);
    }
  };

  const handleDeleteNode = async () => {
    if (!selectedNode) return;
    if (!confirm(`确定删除节点 "${selectedNode.data.name}" 吗？`)) return;

    try {
      await deleteNode(selectedNode.id);
      setNodes((nds) => nds.filter((n) => n.id !== selectedNode.id));
      setEdges((eds) => eds.filter((e) => e.source !== selectedNode.id && e.target !== selectedNode.id));
      setSelectedNode(null);
    } catch (error) {
      console.error('删除节点失败:', error);
    }
  };

  const handleExport = async () => {
    const data: SavedData = {
      worldName,
      worldDescription,
      nodes,
      edges,
      conversations,
      savedAt: new Date().toISOString(),
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${worldName || 'world'}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleImport = async () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';
    input.onchange = async (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (!file) return;
      
      const reader = new FileReader();
      reader.onload = async (e) => {
        try {
          const data = JSON.parse(e.target?.result as string) as SavedData;
          
          setWorldName(data.worldName);
          setWorldDescription(data.worldDescription || '');
          setNodes(data.nodes || []);
          setEdges(data.edges || []);
          setIsWorldLoaded(true);
          
          await createWorld(data.worldName, data.worldDescription || '');
          
          localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
          setLastSaved(new Date().toLocaleTimeString());
          
          console.log('✅ [Import] World imported successfully');
        } catch (error) {
          console.error('导入失败:', error);
        }
      };
      reader.readAsText(file);
    };
    input.click();
  };

  const handleNewWorld = () => {
    if (!confirm('确定要新建世界吗？当前数据将被清除。')) return;
    
    clearLocal();
    setNodes([]);
    setEdges([]);
    setWorldName('');
    setWorldDescription('');
    setSelectedNode(null);
    setShowCreateWorld(true);
  };

  // ========== AI 创建新节点 ==========
  const handleNewNodeFromAI = useCallback((nodeData: { name: string; content: string; rules: string[] }) => {
    const position = {
      x: Math.random() * 400 + 100,
      y: Math.random() * 400 + 100,
    };

    createNode({
      name: nodeData.name,
      node_type: 'custom',
      content: nodeData.content,
      rules: nodeData.rules,
      position_x: position.x,
      position_y: position.y,
      parent_id: selectedNode?.id || null,
    }).then((result) => {
      const newNode: Node = {
        id: result.node.id,
        type: 'worldNode',
        position,
        data: {
          name: result.node.name,
          node_type: result.node.node_type,
          content: result.node.content,
          rules: result.node.rules,
          tags: result.node.tags,
        },
      };
      setNodes((nds) => [...nds, newNode]);
      
      // 选中新创建的节点
      setSelectedNode(newNode);
    }).catch(console.error);
  }, [selectedNode, setNodes]);

  // ========== 沉淀内容到节点 ==========
  const handleSettleContent = useCallback(async (
    content: string, 
    settleType: 'rule' | 'description' | 'subnode',
    targetNodeId?: string
  ) => {
    // 确定目标节点
    let targetNode = targetNodeId 
      ? nodes.find(n => n.id === targetNodeId) 
      : selectedNode;

    // 如果没有目标节点，自动创建一个
    if (!targetNode) {
      const nodeName = prompt('请为新节点命名：', '新设定');
      if (!nodeName) return;

      const position = {
        x: Math.random() * 400 + 100,
        y: Math.random() * 400 + 100,
      };

      try {
        const result = await createNode({
          name: nodeName,
          node_type: 'custom',
          content: settleType === 'description' ? content : '',
          rules: settleType === 'rule' ? [content] : [],
          position_x: position.x,
          position_y: position.y,
          parent_id: null,
        });

        const newNode: Node = {
          id: result.node.id,
          type: 'worldNode',
          position,
          data: {
            name: result.node.name,
            node_type: result.node.node_type,
            content: result.node.content,
            rules: result.node.rules,
            tags: result.node.tags,
          },
        };
        
        setNodes((nds) => [...nds, newNode]);
        // 注意：不再自动选中新节点，避免切换对话导致其他 settleable items 消失
        // setSelectedNode(newNode);
        
        console.log('✅ [Settle] Created new node and settled content');
        return;
      } catch (error) {
        console.error('创建节点失败:', error);
        return;
      }
    }

    // 更新现有节点
    try {
      const currentData = targetNode.data;
      let updatedData = { ...currentData };

      if (settleType === 'rule') {
        // 添加规则
        const currentRules = currentData.rules || [];
        updatedData.rules = [...currentRules, content];
      } else if (settleType === 'description') {
        // 追加描述
        const currentContent = currentData.content || '';
        updatedData.content = currentContent 
          ? `${currentContent}\n\n${content}` 
          : content;
      } else if (settleType === 'subnode') {
        // 创建子节点
        const position = {
          x: (targetNode.position?.x || 0) + 200,
          y: (targetNode.position?.y || 0) + 100,
        };

        const result = await createNode({
          name: content.slice(0, 20) + (content.length > 20 ? '...' : ''),
          node_type: 'custom',
          content: content,
          rules: [],
          position_x: position.x,
          position_y: position.y,
          parent_id: targetNode.id,
        });

        const newNode: Node = {
          id: result.node.id,
          type: 'worldNode',
          position,
          data: {
            name: result.node.name,
            node_type: result.node.node_type,
            content: result.node.content,
            rules: result.node.rules,
            tags: result.node.tags,
          },
        };
        
        setNodes((nds) => [...nds, newNode]);
        setEdges((eds) => [
          ...eds,
          {
            id: `e-${targetNode.id}-${result.node.id}`,
            source: targetNode.id,
            target: result.node.id,
            animated: true,
            style: { stroke: '#6366f1' },
          },
        ]);
        
        console.log('✅ [Settle] Created sub-node');
        return;
      }

      // 更新节点数据（rule 和 description 类型）
      await updateNode(targetNode.id, {
        name: currentData.name,
        node_type: currentData.node_type,
        content: updatedData.content,
        rules: updatedData.rules,
        tags: currentData.tags,
        parent_id: null,
        position_x: targetNode.position?.x || 0,
        position_y: targetNode.position?.y || 0,
      });

      // 更新本地状态
      setNodes((nds) => nds.map((n) => 
        n.id === targetNode!.id 
          ? { ...n, data: updatedData }
          : n
      ));

      // 更新选中节点
      if (selectedNode?.id === targetNode.id) {
        setSelectedNode({ ...targetNode, data: updatedData });
      }

      console.log('✅ [Settle] Updated node:', settleType);
    } catch (error) {
      console.error('❌ [Settle] Failed:', error);
    }
  }, [nodes, selectedNode, setNodes, setEdges]);

  // ========== 对话历史管理 ==========
  
  // 获取当前对话的 key
  // chat 模式：按节点存储（chat_nodeId 或 chat_global）
  // audit/validate/character 模式：全局存储（audit_global, validate_global, character_global）
  const currentConversationKey = (() => {
    if (currentMode === 'chat') {
      return `chat_${selectedNode?.id || 'global'}`;
    }
    // audit, validate, character 模式都是全局的，不按节点区分
    return `${currentMode}_global`;
  })();
  
  // 获取当前对话的历史
  const currentConversation = conversations[currentConversationKey] || [];
  
  // 更新对话历史
  const updateConversation = useCallback((messages: ChatMessage[]) => {
    // 只保留最近的消息
    const trimmedMessages = messages.slice(-MAX_HISTORY_MESSAGES);
    
    setConversations(prev => ({
      ...prev,
      [currentConversationKey]: trimmedMessages,
    }));
  }, [currentConversationKey]);

  // 获取所有节点列表（供 ChatPanel 使用）
  const nodesList = nodes.map(n => ({
    id: n.id,
    name: n.data?.name || '未命名',
  }));

  return (
    <div className="h-screen flex">
      {/* 创建世界弹窗 */}
      {showCreateWorld && (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50">
          <div className="bg-[var(--secondary)] p-8 rounded-2xl border border-[var(--border)] max-w-md w-full mx-4">
            <div className="flex items-center gap-3 mb-6">
              <Globe className="w-8 h-8 text-[var(--primary)]" />
              <h2 className="text-2xl font-bold">创建新世界</h2>
            </div>
            <input
              type="text"
              value={newWorldName}
              onChange={(e) => setNewWorldName(e.target.value)}
              placeholder="世界名称"
              className="input-field mb-4"
              autoFocus
            />
            <textarea
              value={newWorldDesc}
              onChange={(e) => setNewWorldDesc(e.target.value)}
              placeholder="世界简介（可选）"
              className="input-field mb-6"
              rows={3}
            />
            <div className="flex gap-3">
              <button
                onClick={handleCreateWorld}
                disabled={!newWorldName.trim()}
                className="btn-primary flex-1 disabled:opacity-50"
              >
                创建
              </button>
              {isWorldLoaded && (
                <button
                  onClick={() => setShowCreateWorld(false)}
                  className="btn-secondary"
                >
                  取消
                </button>
              )}
            </div>
          </div>
        </div>
      )}

      {/* 编辑节点弹窗 */}
      {showEditNode && editingNode && (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50">
          <div className="bg-[var(--secondary)] p-8 rounded-2xl border border-[var(--border)] max-w-lg w-full mx-4">
            <h2 className="text-2xl font-bold mb-6">编辑节点</h2>
            
            <div className="mb-4">
              <label className="text-sm text-gray-400 mb-2 block">节点名称</label>
              <input
                type="text"
                value={editNodeName}
                onChange={(e) => setEditNodeName(e.target.value)}
                placeholder="节点名称"
                className="input-field"
                autoFocus
              />
            </div>
            
            <div className="mb-4">
              <label className="text-sm text-gray-400 mb-2 block">详细描述</label>
              <textarea
                value={editNodeContent}
                onChange={(e) => setEditNodeContent(e.target.value)}
                placeholder="节点的详细描述..."
                className="input-field"
                rows={5}
              />
            </div>
            
            <div className="mb-6">
              <label className="text-sm text-gray-400 mb-2 block">规则（每行一条）</label>
              <textarea
                value={editNodeRules}
                onChange={(e) => setEditNodeRules(e.target.value)}
                placeholder="每行输入一条规则..."
                className="input-field"
                rows={4}
              />
            </div>
            
            <div className="flex gap-3">
              <button
                onClick={() => {
                  setShowEditNode(false);
                  setEditingNode(null);
                }}
                className="btn-secondary flex-1"
              >
                取消
              </button>
              <button
                onClick={handleSaveEditNode}
                disabled={!editNodeName.trim()}
                className="btn-primary flex-1 disabled:opacity-50"
              >
                保存
              </button>
            </div>
          </div>
        </div>
      )}

      {/* 左侧：画布 */}
      <div className="flex-1 relative">
        {/* 顶部工具栏 - 单行布局 */}
        <div className="absolute top-4 left-4 right-[420px] z-10">
          <div className="flex items-center justify-between gap-4">
            {/* 左侧：世界名称 + 节点操作 */}
            <div className="flex items-center gap-2 flex-shrink-0">
              <div className="bg-[var(--secondary)] px-3 py-2 rounded-lg border border-[var(--border)] flex items-center gap-2">
                <Globe className="w-4 h-4 text-[var(--primary)]" />
                <span className="font-medium text-sm truncate max-w-[120px]">{worldName || '未命名'}</span>
              </div>
              
              <button onClick={handleAddNode} className="btn-secondary flex items-center gap-1 text-sm px-3 py-2">
                <Plus className="w-4 h-4" />
                <span className="hidden sm:inline">添加</span>
              </button>
              
              {selectedNode && (
                <button onClick={handleDeleteNode} className="btn-secondary flex items-center gap-1 text-sm px-3 py-2 text-red-400 border-red-400/30 hover:border-red-400">
                  <Trash2 className="w-4 h-4" />
                  <span className="hidden sm:inline">删除</span>
                </button>
              )}
            </div>

            {/* 右侧：保存状态 + 操作按钮 */}
            <div className="flex items-center gap-2 flex-shrink-0">
              {/* 保存状态 */}
              <div className="bg-[var(--secondary)] px-2 py-2 rounded-lg border border-[var(--border)] flex items-center gap-1 text-xs">
                {saveStatus === 'saved' ? (
                  <>
                    <Cloud className="w-3 h-3 text-green-400" />
                    <span className="text-gray-400 hidden md:inline">{lastSaved}</span>
                  </>
                ) : saveStatus === 'saving' ? (
                  <Cloud className="w-3 h-3 text-yellow-400 animate-pulse" />
                ) : (
                  <CloudOff className="w-3 h-3 text-red-400" />
                )}
              </div>
              
              <button onClick={handleExport} className="btn-secondary flex items-center gap-1 text-sm px-3 py-2">
                <Save className="w-4 h-4" />
                <span className="hidden lg:inline">导出</span>
              </button>
              <button onClick={handleImport} className="btn-secondary flex items-center gap-1 text-sm px-3 py-2">
                <Upload className="w-4 h-4" />
                <span className="hidden lg:inline">导入</span>
              </button>
              <button onClick={handleNewWorld} className="btn-secondary flex items-center gap-1 text-sm px-3 py-2">
                <Plus className="w-4 h-4" />
                <span className="hidden lg:inline">新建</span>
              </button>
            </div>
          </div>
        </div>

        {/* 当前选中节点提示 */}
        {selectedNode && (
          <div className="absolute top-16 left-4 z-10">
            <div className="bg-[var(--accent)]/20 border border-[var(--accent)]/50 px-3 py-1.5 rounded-lg text-sm">
              <span className="text-[var(--accent)]">当前节点:</span>
              <span className="ml-2 text-white font-medium">{selectedNode.data?.name}</span>
            </div>
          </div>
        )}

        {/* React Flow 画布 */}
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={handleNodesChange}
          onEdgesChange={handleEdgesChange}
          onConnect={onConnect}
          onNodeClick={onNodeClick}
          onNodeDoubleClick={onNodeDoubleClick}
          onPaneClick={onPaneClick}
          nodeTypes={nodeTypes}
          fitView
          attributionPosition="bottom-left"
        >
          <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="#2a2a3e" />
          <Controls />
          <MiniMap 
            nodeColor={(node) => {
              const colors: Record<string, string> = {
                world: '#6366f1',
                region: '#10b981',
                race: '#f59e0b',
                character: '#ec4899',
                magic: '#8b5cf6',
                tech: '#06b6d4',
                religion: '#f97316',
                history: '#64748b',
              };
              return colors[node.data?.node_type] || '#6b7280';
            }}
          />
        </ReactFlow>
      </div>

      {/* 右侧：聊天面板 */}
      <div className="w-[400px]">
        <ChatPanel
          selectedNodeId={selectedNode?.id}
          selectedNodeName={selectedNode?.data?.name}
          nodesList={nodesList}
          onNewNode={handleNewNodeFromAI}
          onSettleContent={handleSettleContent}
          initialMessages={currentConversation}
          onMessagesChange={updateConversation}
          currentMode={currentMode}
          onModeChange={setCurrentMode}
          key={currentConversationKey}
        />
      </div>
    </div>
  );
}
