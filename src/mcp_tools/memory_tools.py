from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json

from sqlalchemy import select, func, and_, or_, desc
from loguru import logger

from src.database.connection import db_manager
from src.database.models import ChatMemory


class MemoryTools:
    """MCP tools for chat memory and conversation management."""
    
    async def store_message(
        self,
        session_id: str,
        message_type: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Store a chat message in conversation memory."""
        try:
            # Validate message type
            valid_types = ['user', 'assistant', 'system']
            if message_type not in valid_types:
                return {
                    "error": f"Invalid message type '{message_type}'. Must be one of: {valid_types}",
                    "success": False
                }
            
            # Create chat memory record
            chat_message = ChatMemory(
                session_id=session_id,
                message_type=message_type,
                content=content,
                extra_metadata=metadata or {}
            )
            
            async with db_manager.get_async_session() as session:
                session.add(chat_message)
                await session.commit()
                await session.refresh(chat_message)
            
            logger.info(f"Stored {message_type} message for session {session_id}")
            
            return {
                "success": True,
                "message_id": chat_message.id,
                "session_id": session_id,
                "message_type": message_type,
                "content_length": len(content),
                "created_at": chat_message.created_at.isoformat(),
                "metadata": chat_message.extra_metadata
            }
            
        except Exception as e:
            logger.error(f"Error storing chat message for session {session_id}: {e}")
            return {
                "error": str(e),
                "success": False
            }
    
    async def retrieve_messages(
        self,
        session_id: str,
        limit: int = 50,
        message_types: Optional[List[str]] = None,
        since: Optional[str] = None
    ) -> Dict[str, Any]:
        """Retrieve chat conversation history."""
        try:
            async with db_manager.get_async_session() as session:
                # Build query
                query = select(ChatMemory).where(ChatMemory.session_id == session_id)
                
                # Apply message type filter
                if message_types:
                    query = query.where(ChatMemory.message_type.in_(message_types))
                
                # Apply time filter
                if since:
                    try:
                        since_dt = datetime.fromisoformat(since.replace('Z', '+00:00'))
                        query = query.where(ChatMemory.created_at >= since_dt)
                    except ValueError:
                        return {
                            "error": f"Invalid datetime format for 'since': {since}",
                            "messages": []
                        }
                
                # Order by creation time (oldest first) and apply limit
                query = query.order_by(ChatMemory.created_at.asc()).limit(limit)
                
                # Execute query
                result = await session.execute(query)
                messages = result.scalars().all()
                
                # Format messages
                message_list = []
                for msg in messages:
                    message_list.append({
                        "id": msg.id,
                        "message_type": msg.message_type,
                        "content": msg.content,
                        "metadata": msg.extra_metadata,
                        "created_at": msg.created_at.isoformat(),
                        "content_length": len(msg.content)
                    })
                
                # Get session statistics
                stats_query = select(
                    func.count(ChatMemory.id).label('total_messages'),
                    func.min(ChatMemory.created_at).label('first_message'),
                    func.max(ChatMemory.created_at).label('last_message')
                ).where(ChatMemory.session_id == session_id)
                
                stats_result = await session.execute(stats_query)
                stats = stats_result.first()
                
                return {
                    "session_id": session_id,
                    "messages": message_list,
                    "metadata": {
                        "total_messages_returned": len(message_list),
                        "limit_applied": limit,
                        "filters_applied": {
                            "message_types": message_types,
                            "since": since
                        }
                    },
                    "session_statistics": {
                        "total_messages_in_session": stats.total_messages or 0,
                        "first_message_at": stats.first_message.isoformat() if stats.first_message else None,
                        "last_message_at": stats.last_message.isoformat() if stats.last_message else None
                    }
                }
                
        except Exception as e:
            logger.error(f"Error retrieving messages for session {session_id}: {e}")
            return {
                "error": str(e),
                "messages": []
            }
    
    async def clear_session(
        self,
        session_id: str,
        confirm: bool = False
    ) -> Dict[str, Any]:
        """Clear chat memory for a session."""
        try:
            if not confirm:
                return {
                    "error": "Confirmation required. Set 'confirm' to true to clear session memory.",
                    "success": False,
                    "session_id": session_id
                }
            
            async with db_manager.get_async_session() as session:
                # Count messages before deletion
                count_query = select(func.count(ChatMemory.id)).where(
                    ChatMemory.session_id == session_id
                )
                count_result = await session.execute(count_query)
                message_count = count_result.scalar()
                
                # Delete all messages for the session
                delete_query = select(ChatMemory).where(ChatMemory.session_id == session_id)
                delete_result = await session.execute(delete_query)
                messages_to_delete = delete_result.scalars().all()
                
                for msg in messages_to_delete:
                    await session.delete(msg)
                
                await session.commit()
            
            logger.info(f"Cleared {message_count} messages from session {session_id}")
            
            return {
                "success": True,
                "session_id": session_id,
                "messages_deleted": message_count,
                "cleared_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error clearing session {session_id}: {e}")
            return {
                "error": str(e),
                "success": False
            }
    
    async def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get a summary of a chat session."""
        try:
            async with db_manager.get_async_session() as session:
                # Get session statistics
                stats_query = select(
                    func.count(ChatMemory.id).label('total_messages'),
                    func.count(func.nullif(ChatMemory.message_type != 'user', True)).label('user_messages'),
                    func.count(func.nullif(ChatMemory.message_type != 'assistant', True)).label('assistant_messages'),
                    func.count(func.nullif(ChatMemory.message_type != 'system', True)).label('system_messages'),
                    func.min(ChatMemory.created_at).label('first_message'),
                    func.max(ChatMemory.created_at).label('last_message'),
                    func.sum(func.length(ChatMemory.content)).label('total_content_length')
                ).where(ChatMemory.session_id == session_id)
                
                stats_result = await session.execute(stats_query)
                stats = stats_result.first()
                
                if not stats.total_messages:
                    return {
                        "session_id": session_id,
                        "exists": False,
                        "message": "No messages found for this session"
                    }
                
                # Get recent messages for preview
                recent_query = select(ChatMemory).where(
                    ChatMemory.session_id == session_id
                ).order_by(ChatMemory.created_at.desc()).limit(5)
                
                recent_result = await session.execute(recent_query)
                recent_messages = recent_result.scalars().all()
                
                # Calculate session duration
                duration = None
                if stats.first_message and stats.last_message:
                    duration_delta = stats.last_message - stats.first_message
                    duration = {
                        "total_seconds": duration_delta.total_seconds(),
                        "human_readable": str(duration_delta)
                    }
                
                # Format recent messages preview
                recent_preview = []
                for msg in reversed(recent_messages):  # Reverse to show chronological order
                    recent_preview.append({
                        "message_type": msg.message_type,
                        "content_preview": msg.content[:100] + "..." if len(msg.content) > 100 else msg.content,
                        "created_at": msg.created_at.isoformat()
                    })
                
                return {
                    "session_id": session_id,
                    "exists": True,
                    "statistics": {
                        "total_messages": stats.total_messages,
                        "user_messages": stats.user_messages or 0,
                        "assistant_messages": stats.assistant_messages or 0,
                        "system_messages": stats.system_messages or 0,
                        "total_content_length": stats.total_content_length or 0,
                        "average_message_length": (stats.total_content_length or 0) // (stats.total_messages or 1)
                    },
                    "timeline": {
                        "first_message_at": stats.first_message.isoformat() if stats.first_message else None,
                        "last_message_at": stats.last_message.isoformat() if stats.last_message else None,
                        "duration": duration
                    },
                    "recent_messages_preview": recent_preview
                }
                
        except Exception as e:
            logger.error(f"Error getting session summary for {session_id}: {e}")
            return {
                "error": str(e),
                "session_id": session_id
            }
    
    async def list_active_sessions(
        self,
        hours_back: int = 24,
        limit: int = 20
    ) -> Dict[str, Any]:
        """List recently active chat sessions."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
            
            async with db_manager.get_async_session() as session:
                # Get sessions with activity in the specified timeframe
                sessions_query = select(
                    ChatMemory.session_id,
                    func.count(ChatMemory.id).label('message_count'),
                    func.min(ChatMemory.created_at).label('first_message'),
                    func.max(ChatMemory.created_at).label('last_message')
                ).where(
                    ChatMemory.created_at >= cutoff_time
                ).group_by(
                    ChatMemory.session_id
                ).order_by(
                    func.max(ChatMemory.created_at).desc()
                ).limit(limit)
                
                sessions_result = await session.execute(sessions_query)
                sessions = sessions_result.fetchall()
                
                # Format session list
                active_sessions = []
                for sess in sessions:
                    session_duration = None
                    if sess.first_message and sess.last_message:
                        duration_delta = sess.last_message - sess.first_message
                        session_duration = duration_delta.total_seconds()
                    
                    active_sessions.append({
                        "session_id": sess.session_id,
                        "message_count": sess.message_count,
                        "first_message_at": sess.first_message.isoformat(),
                        "last_message_at": sess.last_message.isoformat(),
                        "duration_seconds": session_duration,
                        "is_recent": (datetime.utcnow() - sess.last_message).total_seconds() < 3600  # Less than 1 hour
                    })
                
                return {
                    "active_sessions": active_sessions,
                    "metadata": {
                        "total_sessions_found": len(active_sessions),
                        "time_window_hours": hours_back,
                        "cutoff_time": cutoff_time.isoformat(),
                        "limit_applied": limit
                    }
                }
                
        except Exception as e:
            logger.error(f"Error listing active sessions: {e}")
            return {
                "error": str(e),
                "active_sessions": []
            }
    
    async def search_conversations(
        self,
        search_text: str,
        session_ids: Optional[List[str]] = None,
        message_types: Optional[List[str]] = None,
        limit: int = 20,
        case_sensitive: bool = False
    ) -> Dict[str, Any]:
        """Search for text across conversation history."""
        try:
            async with db_manager.get_async_session() as session:
                # Build search query
                query = select(ChatMemory)
                
                # Apply text search
                if case_sensitive:
                    search_condition = ChatMemory.content.contains(search_text)
                else:
                    search_condition = ChatMemory.content.ilike(f'%{search_text}%')
                
                query = query.where(search_condition)
                
                # Apply session filter
                if session_ids:
                    query = query.where(ChatMemory.session_id.in_(session_ids))
                
                # Apply message type filter
                if message_types:
                    query = query.where(ChatMemory.message_type.in_(message_types))
                
                # Order by relevance (could be enhanced with full-text search)
                query = query.order_by(ChatMemory.created_at.desc()).limit(limit)
                
                # Execute query
                result = await session.execute(query)
                messages = result.scalars().all()
                
                # Format results with search highlighting
                search_results = []
                for msg in messages:
                    # Simple highlighting - could be enhanced
                    highlighted_content = self._highlight_search_text(
                        msg.content, search_text, case_sensitive
                    )
                    
                    search_results.append({
                        "message_id": msg.id,
                        "session_id": msg.session_id,
                        "message_type": msg.message_type,
                        "content": msg.content,
                        "highlighted_content": highlighted_content,
                        "created_at": msg.created_at.isoformat(),
                        "metadata": msg.metadata
                    })
                
                return {
                    "search_text": search_text,
                    "results": search_results,
                    "metadata": {
                        "total_results": len(search_results),
                        "max_results": limit,
                        "search_options": {
                            "case_sensitive": case_sensitive
                        },
                        "filters_applied": {
                            "session_ids": session_ids,
                            "message_types": message_types
                        }
                    }
                }
                
        except Exception as e:
            logger.error(f"Error searching conversations for '{search_text}': {e}")
            return {
                "error": str(e),
                "search_text": search_text,
                "results": []
            }
    
    def _highlight_search_text(
        self,
        content: str,
        search_text: str,
        case_sensitive: bool,
        highlight_start: str = "**",
        highlight_end: str = "**"
    ) -> str:
        """Highlight search text in content with markdown-style highlighting."""
        if case_sensitive:
            return content.replace(
                search_text,
                f"{highlight_start}{search_text}{highlight_end}"
            )
        else:
            # Case-insensitive replacement
            import re
            pattern = re.escape(search_text)
            return re.sub(
                pattern,
                f"{highlight_start}\\g<0>{highlight_end}",
                content,
                flags=re.IGNORECASE
            )