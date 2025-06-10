import React from 'react';
import SignatureImage from '../assets/signature.png'; // replace with your actual path


const Signature = () => {
  return (
    <div className="flex flex-col items-center justify-center h-screen bg-white px-4">
      {/* خط أفقي */}
      <hr className="w-1/2 border-gray-300 mb-8" />

      {/* صورة التوقيع بحجم أكبر */}
      <img
        src={SignatureImage}
        alt="Signature"
        className="w-64 h-auto object-contain" // <-- كبرنا من w-32 إلى w-64
      />
    </div>
  );
};

export default Signature;