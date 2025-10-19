// Performance optimization hooks and utilities for React components
import React, { useCallback, useMemo, useRef, useEffect } from 'react'

// Debounced state update hook for streaming
export function useStreamingOptimization() {
  const updateQueue = useRef<string[]>([])
  const frameRef = useRef<number>()
  
  const flushUpdates = useCallback((callback: (accumulated: string) => void) => {
    if (frameRef.current) {
      cancelAnimationFrame(frameRef.current)
    }
    
    frameRef.current = requestAnimationFrame(() => {
      if (updateQueue.current.length > 0) {
        const accumulated = updateQueue.current.join('')
        updateQueue.current = []
        callback(accumulated)
      }
    })
  }, [])
  
  const queueUpdate = useCallback((text: string, callback: (accumulated: string) => void) => {
    updateQueue.current.push(text)
    flushUpdates(callback)
  }, [flushUpdates])
  
  useEffect(() => {
    return () => {
      if (frameRef.current) {
        cancelAnimationFrame(frameRef.current)
      }
    }
  }, [])
  
  return { queueUpdate }
}

// Virtual scrolling for large message lists
export function useVirtualScrolling(itemHeight: number, containerHeight: number, items: any[]) {
  const scrollTop = useRef(0)
  const startIndex = Math.floor(scrollTop.current / itemHeight)
  const endIndex = Math.min(startIndex + Math.ceil(containerHeight / itemHeight) + 1, items.length)
  
  const visibleItems = useMemo(() => 
    items.slice(startIndex, endIndex).map((item, index) => ({
      ...item,
      index: startIndex + index
    }))
  , [items, startIndex, endIndex])
  
  return {
    visibleItems,
    totalHeight: items.length * itemHeight,
    offsetY: startIndex * itemHeight
  }
}

// Memoized message component
export const MemoizedMessage = React.memo(({ message, onAction }: any) => {
  return (
    <div className="message-wrapper">
      {/* Message content */}
    </div>
  )
}, (prevProps, nextProps) => {
  return prevProps.message.text === nextProps.message.text &&
         prevProps.message.id === nextProps.message.id
})

// CSS-in-JS performance optimization
export const optimizedStyles = {
  // Use transform instead of changing layout properties
  messageSlideIn: {
    transform: 'translateY(20px)',
    opacity: 0,
    animation: 'slideIn 0.2s ease-out forwards'
  },
  // Hardware acceleration
  hwAccelerated: {
    transform: 'translateZ(0)',
    willChange: 'transform'
  },
  // Reduce paint/layout thrashing
  contentVisibility: {
    contentVisibility: 'auto',
    containIntrinsicSize: '1px 100px'
  }
}