// Simple analytics module for ShortLoom

// Event types for tracking
export enum EventType {
  PAGE_VIEW = 'page_view',
  VIDEO_UPLOAD = 'video_upload',
  YOUTUBE_IMPORT = 'youtube_import',
  CLIP_GENERATED = 'clip_generated',
  CLIP_DOWNLOAD = 'clip_download',
  CLIP_SHARE = 'clip_share',
  USER_REGISTER = 'user_register',
  USER_LOGIN = 'user_login',
  ERROR = 'error'
}

// Analytics event interface
interface AnalyticsEvent {
  type: EventType;
  timestamp: number;
  data?: Record<string, any>;
  userId?: string | null;
}

// Get user ID if available
const getUserId = (): string | null => {
  try {
    const auth = localStorage.getItem('shortloom_auth');
    if (auth) {
      const parsed = JSON.parse(auth);
      // Extract user ID from JWT if available
      // This is a simplified approach - in production you'd use a proper JWT decoder
      return parsed.user_id || null;
    }
    return null;
  } catch (e) {
    return null;
  }
};

// Track an event
export const trackEvent = (type: EventType, data?: Record<string, any>): void => {
  const event: AnalyticsEvent = {
    type,
    timestamp: Date.now(),
    data,
    userId: getUserId()
  };

  // Log to console in development
  if (process.env.NODE_ENV === 'development') {
    console.log('Analytics event:', event);
  }

  // In a real app, you would send this to your analytics service
  // For now, we'll just store it locally
  storeEvent(event);

  // You could also implement a server-side endpoint to collect events
  // sendToServer(event);
};

// Store events locally for demo purposes
const storeEvent = (event: AnalyticsEvent): void => {
  try {
    const events = localStorage.getItem('shortloom_analytics') || '[]';
    const parsedEvents = JSON.parse(events) as AnalyticsEvent[];
    parsedEvents.push(event);
    
    // Keep only the last 100 events to avoid localStorage size issues
    const trimmedEvents = parsedEvents.slice(-100);
    localStorage.setItem('shortloom_analytics', JSON.stringify(trimmedEvents));
  } catch (e) {
    console.error('Failed to store analytics event:', e);
  }
};

// Get all stored events (for debugging/admin purposes)
export const getStoredEvents = (): AnalyticsEvent[] => {
  try {
    const events = localStorage.getItem('shortloom_analytics') || '[]';
    return JSON.parse(events) as AnalyticsEvent[];
  } catch (e) {
    console.error('Failed to retrieve analytics events:', e);
    return [];
  }
};

// Clear stored events
export const clearStoredEvents = (): void => {
  localStorage.removeItem('shortloom_analytics');
};

// Track page views automatically
export const initAnalytics = (): void => {
  // Track initial page view
  trackEvent(EventType.PAGE_VIEW, { path: window.location.pathname });

  // Track route changes if using a router
  // This is a simplified example - adapt to your routing library
  const originalPushState = history.pushState;
  history.pushState = function(...args) {
    const result = originalPushState.apply(this, args);
    trackEvent(EventType.PAGE_VIEW, { path: window.location.pathname });
    return result;
  };

  // Track errors
  window.addEventListener('error', (e) => {
    trackEvent(EventType.ERROR, { 
      message: e.message,
      source: e.filename,
      lineno: e.lineno,
      colno: e.colno
    });
  });

  return;
};
