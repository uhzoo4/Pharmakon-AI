## UI Pro Max Stack Guidelines
**Stack:** react | **Query:** SSE realtime updates list rerender performance
**Source:** stacks/react.csv | **Found:** 4 results

### Result 1
- **Category:** Performance
- **Guideline:** Batch state updates
- **Description:** React 18 auto-batches but be aware
- **Do:** Let React batch related updates
- **Don't:** Manual batching with flushSync
- **Code Good:** setA(1); setB(2); // batched
- **Code Bad:** flushSync(() => setA(1))
- **Severity:** Low
- **Docs URL:** https://react.dev/learn/queueing-a-series-of-state-updates

### Result 2
- **Category:** Accessibility
- **Guideline:** Announce dynamic content
- **Description:** Use ARIA live regions for updates
- **Do:** aria-live for dynamic updates
- **Don't:** Silent updates to screen readers
- **Code Good:** <div aria-live="polite">{msg}</div>
- **Code Bad:** <div>{msg}</div>
- **Severity:** Medium
- **Docs URL:** 

### Result 3
- **Category:** Performance
- **Guideline:** Use React DevTools Profiler
- **Description:** Profile to identify performance bottlenecks
- **Do:** Profile before optimizing
- **Don't:** Optimize without measuring
- **Code Good:** React DevTools Profiler
- **Code Bad:** Guessing at bottlenecks
- **Severity:** Medium
- **Docs URL:** https://react.dev/learn/react-developer-tools

### Result 4
- **Category:** TypeScript
- **Guideline:** Use generics for reusable components
- **Description:** Generic components for flexible typing
- **Do:** Generic props for list components
- **Don't:** Union types for flexibility
- **Code Good:** <List<T> items={T[]}>
- **Code Bad:** <List items={any[]}>
- **Severity:** Medium
- **Docs URL:** 

