'use client';

import { memo, useState } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { 
  Globe, MapPin, Users, User, Sparkles, Cpu, 
  Church, Clock, Building, ScrollText, Package, Bug, Folder
} from 'lucide-react';

// 节点类型到图标的映射
const typeIcons: Record<string, React.ReactNode> = {
  world: <Globe className="w-4 h-4" />,
  region: <MapPin className="w-4 h-4" />,
  race: <Users className="w-4 h-4" />,
  character: <User className="w-4 h-4" />,
  magic: <Sparkles className="w-4 h-4" />,
  tech: <Cpu className="w-4 h-4" />,
  religion: <Church className="w-4 h-4" />,
  history: <Clock className="w-4 h-4" />,
  org: <Building className="w-4 h-4" />,
  rule: <ScrollText className="w-4 h-4" />,
  item: <Package className="w-4 h-4" />,
  creature: <Bug className="w-4 h-4" />,
  custom: <Folder className="w-4 h-4" />,
};

// 节点类型到颜色的映射
const typeColors: Record<string, string> = {
  world: '#6366f1',
  region: '#10b981',
  race: '#f59e0b',
  character: '#ec4899',
  magic: '#8b5cf6',
  tech: '#06b6d4',
  religion: '#f97316',
  history: '#64748b',
  org: '#84cc16',
  rule: '#ef4444',
  item: '#eab308',
  creature: '#14b8a6',
  custom: '#6b7280',
};

// 节点类型中文名
const typeNames: Record<string, string> = {
  world: '世界',
  region: '地区',
  race: '种族',
  character: '角色',
  magic: '魔法',
  tech: '科技',
  religion: '宗教',
  history: '历史',
  org: '组织',
  rule: '规则',
  item: '物品',
  creature: '生物',
  custom: '自定义',
};

interface WorldNodeData {
  name: string;
  node_type: string;
  content: string;
  rules: string[];
  tags: string[];
}

function WorldNodeComponent({ data, selected }: NodeProps<WorldNodeData>) {
  const nodeType = data.node_type || 'custom';
  const icon = typeIcons[nodeType] || typeIcons.custom;
  const color = typeColors[nodeType] || typeColors.custom;
  const typeName = typeNames[nodeType] || '自定义';
  
  // Tooltip 状态
  const [showTooltip, setShowTooltip] = useState(false);
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });

  const handleMouseEnter = (e: React.MouseEvent) => {
    const rect = e.currentTarget.getBoundingClientRect();
    setTooltipPos({ x: rect.width + 10, y: 0 });
    setShowTooltip(true);
  };

  const handleMouseLeave = () => {
    setShowTooltip(false);
  };

  return (
    <div 
      className={`world-node ${selected ? 'selected' : ''}`}
      style={{ borderColor: selected ? '#f59e0b' : color }}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      {/* 输入连接点 */}
      <Handle
        type="target"
        position={Position.Top}
        style={{ 
          background: color,
          width: 10,
          height: 10,
          border: '2px solid #1e1e2e'
        }}
      />
      
      {/* 节点头部 */}
      <div className="flex items-center gap-2 mb-2">
        <span style={{ color }}>{icon}</span>
        <span className="node-type" style={{ color }}>{typeName}</span>
      </div>
      
      {/* 节点标题 */}
      <div className="node-title">{data.name}</div>
      
      {/* 节点内容预览 */}
      {data.content && (
        <div className="node-content">{data.content}</div>
      )}
      
      {/* 规则标签 */}
      {data.rules && data.rules.length > 0 && (
        <div className="flex flex-wrap gap-1 mt-2">
          {data.rules.slice(0, 2).map((rule, i) => (
            <span 
              key={i}
              className="text-xs px-2 py-0.5 rounded-full"
              style={{ background: `${color}20`, color }}
            >
              {rule.length > 15 ? rule.slice(0, 15) + '...' : rule}
            </span>
          ))}
          {data.rules.length > 2 && (
            <span className="text-xs text-gray-500">+{data.rules.length - 2}</span>
          )}
        </div>
      )}
      
      {/* 输出连接点 */}
      <Handle
        type="source"
        position={Position.Bottom}
        style={{ 
          background: color,
          width: 10,
          height: 10,
          border: '2px solid #1e1e2e'
        }}
      />
      
      {/* Tooltip - 悬停显示详细信息 */}
      {showTooltip && (data.content || (data.rules && data.rules.length > 0)) && (
        <div 
          className="absolute z-50 bg-[#1e1e2e] border border-[#3a3a4e] rounded-lg p-4 shadow-xl min-w-[280px] max-w-[400px]"
          style={{ 
            left: tooltipPos.x, 
            top: tooltipPos.y,
            pointerEvents: 'none'
          }}
        >
          {/* 标题 */}
          <div className="flex items-center gap-2 mb-3 pb-2 border-b border-[#3a3a4e]">
            <span style={{ color }}>{icon}</span>
            <span className="font-bold text-white">{data.name}</span>
            <span className="text-xs px-2 py-0.5 rounded" style={{ background: `${color}30`, color }}>
              {typeName}
            </span>
          </div>
          
          {/* 完整内容 */}
          {data.content && (
            <div className="mb-3">
              <div className="text-xs text-gray-400 mb-1">描述</div>
              <div className="text-sm text-gray-200 whitespace-pre-wrap">{data.content}</div>
            </div>
          )}
          
          {/* 所有规则 */}
          {data.rules && data.rules.length > 0 && (
            <div>
              <div className="text-xs text-gray-400 mb-1">规则 ({data.rules.length})</div>
              <div className="flex flex-col gap-1">
                {data.rules.map((rule, i) => (
                  <div 
                    key={i}
                    className="text-xs px-2 py-1 rounded"
                    style={{ background: `${color}15`, color }}
                  >
                    • {rule}
                  </div>
                ))}
              </div>
            </div>
          )}
          
          {/* 提示 */}
          <div className="mt-3 pt-2 border-t border-[#3a3a4e] text-xs text-gray-500">
            💡 双击编辑此节点
          </div>
        </div>
      )}
    </div>
  );
}

export default memo(WorldNodeComponent);

