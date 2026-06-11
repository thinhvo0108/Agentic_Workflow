import { createIcon } from '@chakra-ui/react';

const BotIcon = createIcon({
  displayName: 'BotIcon',
  viewBox: '0 0 24 24',
  path: (
    <>
      {/* Head */}
      <rect x="3" y="7" width="18" height="13" rx="2" stroke="currentColor" strokeWidth="2" fill="none" />
      {/* Eyes */}
      <circle cx="9" cy="13" r="1.5" fill="currentColor" />
      <circle cx="15" cy="13" r="1.5" fill="currentColor" />
      {/* Mouth */}
      <path d="M9 17h6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
      {/* Antenna */}
      <path d="M12 7V4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
      <circle cx="12" cy="3" r="1" fill="currentColor" />
    </>
  ),
});

export default BotIcon;
