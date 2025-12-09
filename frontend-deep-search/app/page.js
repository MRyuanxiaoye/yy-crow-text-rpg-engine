'use client';

import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { useState, useEffect, useRef } from 'react';
import { Send, Sparkles, BookOpen, Search, Mic, ArrowUp, Square, Loader2, Copy, Check } from 'lucide-react';
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function Chat() {
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [explanations, setExplanations] = useState([]); // Array of explanation objects
  const explanationRefs = useRef({}); // Refs for explanation items to scroll to
  const bottomRef = useRef(null);
  const inputRef = useRef(null);
  const abortControllerRef = useRef(null);
  const termCache = useRef({}); // Cache for definitions
  
  // Selection popup state
  const [selectionPopup, setSelectionPopup] = useState({ show: false, x: 0, y: 0, text: '' });

  // Auto-scroll
  useEffect(() => {
    if (messages.length > 0 && bottomRef.current) {
        bottomRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, isLoading]);

  // Handle Text Selection
  useEffect(() => {
    const handleSelection = () => {
        const selection = window.getSelection();
        const selectedText = selection.toString().trim();

        if (selectedText && selectedText.length > 0 && selectedText.length < 50) {
            const range = selection.getRangeAt(0);
            const rect = range.getBoundingClientRect();
            
            setSelectionPopup({
                show: true,
                x: rect.left,
                y: rect.bottom + window.scrollY + 10, // Show below the text
                text: selectedText
            });
        } else {
            setSelectionPopup(prev => ({ ...prev, show: false }));
        }
    };

    document.addEventListener('selectionchange', handleSelection);
    return () => document.removeEventListener('selectionchange', handleSelection);
  }, []);

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!query.trim()) return;

        const userMsg = { role: 'user', content: query };
        setMessages(prev => [...prev, userMsg]);
        const currentQuery = query;
        setQuery('');
        setIsLoading(true);
        setIsStreaming(true);

        // Create new AbortController
        abortControllerRef.current = new AbortController();
        const signal = abortControllerRef.current.signal;

        // Connect to SSE stream (Native EventSource doesn't support signal directly easily, 
        // but we can close it manually on abort)
        const eventSource = new EventSource(`${API_BASE_URL}/api/stream?query=${encodeURIComponent(currentQuery)}`);
        
        // Temporary accumulator
        let currentContent = "";
        
        // Initialize assistant message with logs array
        setMessages(prev => [...prev, { role: 'assistant', content: "", logs: [] }]);

        eventSource.addEventListener("log", (event) => {
             const data = JSON.parse(event.data);
             setMessages(prev => {
                const newMsgs = [...prev];
                if (newMsgs.length === 0) return newMsgs;

                const lastMsgIndex = newMsgs.length - 1;
                // CRITICAL FIX: Create a shallow copy of the message object
                // AND a shallow copy of the logs array to avoid mutating state directly.
                const lastMsg = { ...newMsgs[lastMsgIndex] };
                
                const currentLogs = lastMsg.logs ? [...lastMsg.logs] : [];
                
                // Optional: Front-end deduplication logic
                // Only add if it's not the exact same as the last log (or check entire list)
                // For now, let's trust the backend stream is clean, but fix the React Mutation issue.
                currentLogs.push(data.content);
                
                lastMsg.logs = currentLogs;
                newMsgs[lastMsgIndex] = lastMsg;
                
                return newMsgs;
            });
        });

        eventSource.addEventListener("report", (event) => {
            const data = JSON.parse(event.data);
            
            if (data.full_content) {
                // If backend sends full content marker or keywords at the end
                if (data.keywords) {
                     setMessages(prev => {
                        const newMsgs = [...prev];
                        if (newMsgs.length === 0) return newMsgs;

                        const lastMsgIndex = newMsgs.length - 1;
                        const lastMsg = { ...newMsgs[lastMsgIndex] };
                        
                        lastMsg.keywords = data.keywords;
                        
                        // --- REMOVED HIGHLIGHTING LOGIC ---
                        // We rely on manual selection now.

                        newMsgs[lastMsgIndex] = lastMsg;
                        return newMsgs;
                    });
                }
            } else {
                // Append chunk
                currentContent += data.content;
                setMessages(prev => {
                    const newMsgs = [...prev];
                    if (newMsgs.length === 0) return newMsgs;

                    const lastMsgIndex = newMsgs.length - 1;
                    const lastMsg = { ...newMsgs[lastMsgIndex] };
                    
                    lastMsg.content = currentContent;
                    
                    newMsgs[lastMsgIndex] = lastMsg;
                    return newMsgs;
                });
            }
        });

        eventSource.addEventListener("end", (event) => {
            eventSource.close();
            setIsLoading(false);
            setIsStreaming(false);
        });

        eventSource.onerror = (err) => {
            console.error("Stream error:", err);
            eventSource.close();
            setIsLoading(false);
            setIsStreaming(false);
        };
        
        // Store eventSource in ref to allow stopping
        abortControllerRef.current.eventSource = eventSource;
    };

    const handleStop = () => {
        if (abortControllerRef.current) {
            if (abortControllerRef.current.eventSource) {
                abortControllerRef.current.eventSource.close();
            }
            abortControllerRef.current.abort();
            setIsLoading(false);
            setIsStreaming(false);
            
            // Add a system message or marker?
            setMessages(prev => {
                const newMsgs = [...prev];
                const lastMsg = newMsgs[newMsgs.length - 1];
                if (lastMsg.role === 'assistant') {
                    lastMsg.content += " [Stopped]";
                }
                return newMsgs;
            });
        }
    };

  const handleExplain = async (term) => {
    console.log("Explaining:", term);
    
    // Clear selection popup
    setSelectionPopup(prev => ({ ...prev, show: false }));
    window.getSelection().removeAllRanges();

    // Check if already in list
    const existingIndex = explanations.findIndex(e => e.term === term);
    if (existingIndex !== -1) {
        // Scroll to it
        if (explanationRefs.current[term]) {
             explanationRefs.current[term].scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
        return;
    }

    // Add new item with loading state
    setExplanations(prev => [...prev, { term, definition: null, isLoading: true }]);

    // 1. Check cache first
    if (termCache.current[term]) {
        setExplanations(prev => prev.map(e => 
            e.term === term ? { ...e, definition: termCache.current[term], isLoading: false } : e
        ));
         // Wait for render then scroll
         setTimeout(() => {
             if (explanationRefs.current[term]) {
                 explanationRefs.current[term].scrollIntoView({ behavior: 'smooth', block: 'start' });
             }
         }, 100);
        return;
    }

    try {
        const res = await fetch(`${API_BASE_URL}/api/explain`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ term })
        });
        const data = await res.json();
        
        // 2. Save to cache
        termCache.current[term] = data.definition;
        
        // 3. Update UI
        setExplanations(prev => prev.map(e => 
            e.term === term ? { ...e, definition: data.definition, isLoading: false } : e
        ));
        
         // Wait for render then scroll
         setTimeout(() => {
             if (explanationRefs.current[term]) {
                 explanationRefs.current[term].scrollIntoView({ behavior: 'smooth', block: 'start' });
             }
         }, 100);

    } catch (e) {
        console.error("Explain error:", e);
        setExplanations(prev => prev.map(e => 
            e.term === term ? { ...e, definition: "无法获取解释。", isLoading: false } : e
        ));
    }
  };

  const copyToClipboard = async (text) => {
      try {
          await navigator.clipboard.writeText(text);
          // Optional: Show a toast or temporary success state
          alert("Copied to clipboard!");
      } catch (err) {
          console.error("Failed to copy:", err);
      }
  };

  return (
    <div className="flex h-screen bg-[#212121] text-gray-100 font-sans">
      {/* Selection Popup */}
      {selectionPopup.show && (
          <div 
            className="fixed z-50 animate-in fade-in zoom-in duration-200"
            style={{ top: selectionPopup.y, left: selectionPopup.x }}
          >
              <button 
                onClick={() => handleExplain(selectionPopup.text)}
                className="flex items-center gap-1.5 bg-[#333] hover:bg-[#444] text-white px-3 py-1.5 rounded-lg shadow-xl border border-[#555] text-sm font-medium transition-colors"
              >
                  <Search className="w-3.5 h-3.5" />
                  Explain "{selectionPopup.text}"
              </button>
          </div>
      )}

      {/* Main Content */}
      <div className="flex-1 flex flex-col h-full relative">
        
        {/* Header / Top Bar */}
        <div className="absolute top-4 left-4 z-10">
          <div className="text-lg font-semibold text-gray-300 flex items-center gap-2">
            <span className="bg-white text-black text-xs font-bold px-2 py-0.5 rounded">Deep</span>
            Search
          </div>
        </div>

        {/* Chat Area */}
        <div className="flex-1 overflow-y-auto scrollbar-thin scrollbar-thumb-gray-700">
          {messages.length === 0 ? (
            // Empty State / Welcome Screen
            <div className="h-full flex flex-col items-center justify-center px-4">
              <div className="mb-8 p-4 bg-[#2f2f2f] rounded-full">
                <Sparkles className="w-8 h-8 text-white" />
              </div>
              <h1 className="text-3xl font-medium text-white mb-2">Where should we begin?</h1>
              <p className="text-gray-500 mb-10">Deep research agent at your service.</p>
              
              {/* Suggestion Chips */}
              <div className="flex gap-3 flex-wrap justify-center max-w-2xl">
                {["Explain Quantum Computing", "History of Rome", "Latest AI Trends", "How do engines work?"].map((suggestion) => (
                  <button 
                    key={suggestion}
                    onClick={() => setQuery(suggestion)}
                    className="px-4 py-2 bg-[#2f2f2f] hover:bg-[#424242] rounded-xl text-sm text-gray-300 transition-colors border border-transparent hover:border-gray-600"
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            // Message List
            <div className="flex flex-col items-center py-10 pb-64">
              {messages.map((msg, idx) => (
                <div key={idx} className="w-full max-w-3xl px-4 py-6 group">
                  <div className="flex gap-4">
                    {/* Avatar */}
                    <div className={`w-8 h-8 rounded-sm flex-shrink-0 flex items-center justify-center ${
                      msg.role === 'user' ? 'bg-transparent' : 'bg-green-500/10'
                    }`}>
                      {msg.role === 'user' ? (
                        <div className="w-8 h-8 bg-gray-600 rounded-full flex items-center justify-center text-sm">U</div>
                      ) : (
                        msg.content ? <Sparkles className="w-5 h-5 text-green-400" /> : <div className="animate-spin w-4 h-4 border-2 border-green-400 border-t-transparent rounded-full" />
                      )}
                    </div>

                    {/* Content */}
                    <div className="flex-1 space-y-2 overflow-hidden">
                      <div className="font-medium text-sm text-gray-400 mb-1">
                        {msg.role === 'user' ? 'You' : (msg.content ? 'Deep Search' : '')}
                      </div>
                      <div className="prose prose-invert prose-p:leading-relaxed prose-pre:bg-[#2f2f2f] max-w-none text-gray-100">
                        {msg.role === 'assistant' ? (
                          msg.content ? (
                            <>
                                <ReactMarkdown 
                                remarkPlugins={[remarkGfm]}
                                >
                                {msg.content}
                                </ReactMarkdown>
                                
                                {/* Copy Button */}
                                <div className="mt-4 flex justify-end">
                                    <button 
                                        onClick={() => copyToClipboard(msg.content)}
                                        className="flex items-center gap-1.5 text-xs text-gray-500 hover:text-gray-300 transition-colors px-2 py-1 rounded hover:bg-[#2f2f2f]"
                                    >
                                        <Copy className="w-3.5 h-3.5" />
                                        Copy Answer
                                    </button>
                                </div>
                            </>
                          ) : (
                             /* Thinking State */
                             <div className="space-y-3 animate-in fade-in duration-300">
                                <div className="flex items-center gap-2 text-green-400">
                                    <Sparkles className="w-4 h-4 animate-pulse" />
                                    <span className="font-medium text-sm">Thinking...</span>
                                </div>
                                <div className="text-gray-400 text-sm space-y-1 font-mono pl-6 border-l-2 border-[#2f2f2f]">
                                    {msg.logs && msg.logs.map((log, i) => (
                                        <div key={i} className="animate-in slide-in-from-left-2 duration-300">
                                            {log}
                                        </div>
                                    ))}
                                    <div className="animate-pulse opacity-50">...</div>
                                </div>
                             </div>
                          )
                        ) : (
                          msg.content
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
              {/* Remove bottom thinking indicator */}
              <div ref={bottomRef} />
            </div>
          )}
        </div>

        {/* Input Area (Floating) */}
        <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-[#212121] via-[#212121] to-transparent pt-10 pb-6 px-4">
          <div className="max-w-3xl mx-auto">
            <div className="bg-[#2f2f2f] rounded-[26px] p-2 border border-[#424242] shadow-lg relative flex flex-col transition-colors focus-within:border-gray-500">
              <textarea
                ref={inputRef}
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSubmit(e);
                  }
                }}
                placeholder="Ask anything..."
                className="w-full bg-transparent text-white placeholder-gray-500 resize-none focus:outline-none px-4 py-3 min-h-[52px] max-h-[200px]"
                rows={1}
                style={{ height: 'auto', overflow: 'hidden' }}
              />
              
              {/* Toolbar */}
              <div className="flex justify-between items-center px-2 pb-1">
                <div className="flex gap-2">
                  <button className="p-2 text-gray-400 hover:text-white hover:bg-[#424242] rounded-full transition-colors" title="Search">
                    <Search className="w-4 h-4" />
                  </button>
                </div>
                <button 
                  onClick={isStreaming ? handleStop : handleSubmit}
                  disabled={(!query.trim() && !isStreaming)}
                  className={`p-2 rounded-full transition-all duration-200 ${
                    (query.trim() || isStreaming)
                      ? 'bg-white text-black hover:bg-gray-200' 
                      : 'bg-[#424242] text-gray-500 cursor-not-allowed'
                  }`}
                >
                  {isStreaming ? <Square className="w-5 h-5 fill-current" /> : <ArrowUp className="w-5 h-5" />}
                </button>
              </div>
            </div>
            <div className="text-center text-xs text-gray-500 mt-2">
              Deep Search can make mistakes. Check important info.
            </div>
          </div>
        </div>
      </div>

      {/* Right Sidebar (Explanation) */}
      {explanations.length > 0 && (
        <div className="w-[350px] bg-[#171717] border-l border-[#2f2f2f] shadow-2xl flex flex-col animate-in slide-in-from-right duration-300">
          <div className="p-4 border-b border-[#2f2f2f] flex justify-between items-center bg-[#171717] z-10 sticky top-0">
            <h3 className="font-medium text-gray-200 flex items-center gap-2">
              <BookOpen className="w-4 h-4" /> 
              Dictionary
            </h3>
            <button onClick={() => setExplanations([])} className="text-gray-500 hover:text-white transition-colors">×</button>
          </div>
          <div className="p-0 overflow-y-auto flex-1 pb-10">
            {explanations.map((expl, idx) => (
                <div 
                    key={expl.term} 
                    ref={el => explanationRefs.current[expl.term] = el}
                    className={`p-6 border-b border-[#2f2f2f]/50 ${idx % 2 === 0 ? 'bg-transparent' : 'bg-[#1a1a1a]'}`}
                >
                    <div className="mb-1 text-xs font-bold text-blue-400 uppercase tracking-wider">Term</div>
                    <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
                        {expl.term}
                        {expl.isLoading && <Loader2 className="w-5 h-5 animate-spin text-gray-500" />}
                    </h2>
                    
                    <div className="prose prose-invert prose-sm text-gray-300 leading-relaxed">
                        {expl.isLoading ? (
                            <div className="space-y-2 animate-pulse">
                                <div className="h-4 bg-gray-700 rounded w-3/4"></div>
                                <div className="h-4 bg-gray-700 rounded w-full"></div>
                                <div className="h-4 bg-gray-700 rounded w-5/6"></div>
                            </div>
                        ) : (
                            <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                {expl.definition || ""}
                            </ReactMarkdown>
                        )}
                    </div>
                </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
