import React from 'react';
import { assets } from "../assets/assets";
import Partners_logo1 from '../assets/Partners_logo1.png';
import Partners_logo2 from '../assets/Partners_logo2.png';
import Partners_logo3 from '../assets/Partners_logo3.png';
import pngwing from '../assets/pngwing.png';
import search from '../assets/search.png';

const Partners = () => {
return ( 
<div className="bg-white py-16 text-center "> <h2 className="text-gray-700 text-3xl font-semibold mb-4 tracking-widest">OUR PARTNERS</h2>

  <div className="flex justify-center items-center gap-8 px-4 py-6 bg-primary/90 rounded-lg shadow-md max-w-4xl mx-auto">
    <img src={Partners_logo1} alt="Partner 1" className="h-10 object-contain transition-transform duration-300 hover:scale-110" />
    <img src={Partners_logo2} alt="Partner 2" className="h-10 object-contain transition-transform duration-300 hover:scale-110" />
    <img src={Partners_logo3} alt="Partner 3" className="h-10 object-contain transition-transform duration-300 hover:scale-110" />
    <img src={pngwing} alt="Partner 3" className="h-10 object-contain transition-transform duration-300 hover:scale-110" />
  </div>

  <div className="mt-12 flex flex-col items-center md:flex-row md:justify-center gap-8 px-6">
    <img src={search} alt="search" className="w-32 h-auto" />
    <div className="text-left max-w-lg">
      <h3 className="text-3xl font-bold text-gray-900 mb-4">
        How Accurate is Our AI Tumor Detection?
      </h3>
      <p className="text-gray-600 text-sm">
        Our AI tool uses high-quality X-rays to detect tumors early, improving treatment success.
        Better X-rays mean more accurate results. The tool reduces false positives and negatives,
        ensuring reliable cancer detection.
      </p>
    </div>
  </div>
</div>
);
};

export default Partners;