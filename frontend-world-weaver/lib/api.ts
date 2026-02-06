/**
 * World Weaver API Client
 */

const API_BASE = process.env.NODE_ENV === 'production' 
  ? '/api/world-weaver'
  : 'http://localhost:8001';

// ========== Types ==========

export interface WorldNode {
  id: string;
  name: string;
  node_type: string;
  content: string;
  rules: string[];
  tags: string[];
  parent_id: string | null;
  position_x: number;
  position_y: number;
  created_at: string;
  updated_at: string;
}

export interface NodeRelation {
  id: string;
  source_id: string;
  target_id: string;
  relation_type: string;
  description: string;
}

export interface ChatResponse {
  response: string;
  suggestions: string[];
  conflicts: string[];
  new_nodes: WorldNode[];
}

export interface WorldInfo {
  id: string;
  name: string;
  description: string;
  node_count: number;
  relation_count: number;
}

// ========== API Functions ==========

async function fetchAPI<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const url = `${API_BASE}${endpoint}`;
  const response = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  });
  
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `API Error: ${response.status}`);
  }
  
  return response.json();
}

// ========== World API ==========

export async function createWorld(name: string, description: string = ''): Promise<{ message: string; world_id: string }> {
  return fetchAPI('/world/create', {
    method: 'POST',
    body: JSON.stringify({ name, description }),
  });
}

export async function getWorld(): Promise<WorldInfo> {
  return fetchAPI('/world');
}

export async function exportWorld(): Promise<Record<string, unknown>> {
  return fetchAPI('/world/export');
}

export async function importWorld(data: Record<string, unknown>): Promise<{ message: string }> {
  return fetchAPI('/world/import', {
    method: 'POST',
    body: JSON.stringify({ data }),
  });
}

// ========== Nodes API ==========

export async function getNodes(): Promise<{ nodes: WorldNode[] }> {
  return fetchAPI('/nodes');
}

export async function getNode(nodeId: string): Promise<WorldNode> {
  return fetchAPI(`/nodes/${nodeId}`);
}

export async function createNode(node: {
  name: string;
  node_type?: string;
  content?: string;
  rules?: string[];
  tags?: string[];
  parent_id?: string | null;
  position_x?: number;
  position_y?: number;
}): Promise<{ message: string; node: WorldNode }> {
  return fetchAPI('/nodes', {
    method: 'POST',
    body: JSON.stringify(node),
  });
}

export async function updateNode(nodeId: string, node: {
  name: string;
  node_type?: string;
  content?: string;
  rules?: string[];
  tags?: string[];
  parent_id?: string | null;
  position_x?: number;
  position_y?: number;
}): Promise<{ message: string; node: WorldNode }> {
  return fetchAPI(`/nodes/${nodeId}`, {
    method: 'PUT',
    body: JSON.stringify(node),
  });
}

export async function deleteNode(nodeId: string): Promise<{ message: string }> {
  return fetchAPI(`/nodes/${nodeId}`, {
    method: 'DELETE',
  });
}

// ========== Relations API ==========

export async function getRelations(): Promise<{ relations: NodeRelation[] }> {
  return fetchAPI('/relations');
}

export async function createRelation(relation: {
  source_id: string;
  target_id: string;
  relation_type?: string;
  description?: string;
}): Promise<{ message: string; relation: NodeRelation }> {
  return fetchAPI('/relations', {
    method: 'POST',
    body: JSON.stringify(relation),
  });
}

// ========== Chat API ==========

export interface HistoryMessage {
  role: 'user' | 'assistant';
  content: string;
}

export async function chat(
  message: string, 
  mode: 'audit' | 'expand' | 'validate' | 'chat' | 'character' = 'chat',
  nodeId?: string,
  history?: HistoryMessage[]
): Promise<ChatResponse> {
  return fetchAPI('/chat', {
    method: 'POST',
    body: JSON.stringify({
      message,
      mode,
      node_id: nodeId,
      history: history || [],
    }),
  });
}

export async function validateSetting(
  setting: string,
  validateType: 'science' | 'mythology' | 'logic' = 'science',
  nodeId?: string
): Promise<{ setting: string; validation: string; suggestions: string[] }> {
  return fetchAPI('/validate', {
    method: 'POST',
    body: JSON.stringify({
      setting,
      validate_type: validateType,
      node_id: nodeId,
    }),
  });
}

// ========== Context API ==========

export async function getContext(nodeId: string): Promise<{ context: string }> {
  return fetchAPI(`/context/${nodeId}`);
}

// ========== Health Check ==========

export async function healthCheck(): Promise<{
  status: string;
  models: { openai: boolean; deepseek: boolean; anthropic: boolean };
  world_loaded: boolean;
}> {
  return fetchAPI('/health');
}

